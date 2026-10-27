// ============================================================================
//  snn_paper.v - Cue-conditioned spiking controller (after Xue et al. 2014,
//                "FPGA Implementation of Self-Organized SNN Controller for Mobile Robots")
//
//  STRUCTURE
//    9 sensory neurons : 0 LO  1 LI  2 F  3 RI  4 RO      (distance sensors)
//                        5 CUE_RED  6 CUE_GREEN           (colour sensors)
//                        7 GOAL_L   8 GOAL_R              (goal direction)
//    2*NH hidden       : group H1 (hears CUE_RED), group H2 (hears CUE_GREEN)
//    2 motor neurons   : ML = left wheel, MR = right wheel
//
//  UNCONDITIONED REFLEX (fixed wires, never learns)
//    LO,LI -> ML      RI,RO -> MR      (obstacle on one side -> turn away)
//    F     -> inhibits ML and MR       (obstacle ahead -> slow down)
//    ML <-> MR inhibit each other      (one wheel wins -> a clear turn decision)
//    GOAL_L -> MR     GOAL_R -> ML     (weak steering to the goal)
//    tonic background current into both motors = constant forward drive
//    small LFSR noise in both motors          = random choice when counts tie
//
//  CONDITIONED REFLEX (plastic): hidden -> motor, 2*NH*2 synapses
//    Pair-based STDP, multiplicative (change proportional to the weight), depression
//    slightly larger than potentiation (A- = 1.0625 A+), weights capped.
//    DOPAMINE (input `da`) decides whether STDP is kept:
//      STDP does not change the weight directly, it writes a decaying ELIGIBILITY TAG
//      (+ for pre-before-post, - for post-before-pre). While da=1 a fraction of every tag
//      is turned into a weight change. So a pairing that is followed shortly afterwards by
//      reward is strengthened, a pairing that is never rewarded leaves no trace.
//      Plastic synaptic current is also boosted ~15% while da=1 (paper: 1 + beta*dDA).
//
//  Motor decode (done in the testbench): spike counts of ML / MR over one window.
//    ML > MR -> right wheel slower -> turn RIGHT ;  MR > ML -> turn LEFT.
// ============================================================================
`timescale 1ns/1ps

module snn_paper #(
    parameter TICK_DIV     = 4,
    parameter NH           = 6,        // hidden neurons per cue group
    parameter THR_IN       = 100,
    parameter THR_H        = 100,      // hidden thresholds are THR_H + 7*(h%5)  (heterogeneous)
    parameter THR_MOT      = 200,
    parameter LEAK_H       = 2,
    parameter LEAK_M       = 4,
    parameter W_CUE        = 70,       // cue neuron -> hidden neuron (fixed)
    parameter W_REFLEX     = 90,       // distance sensor -> motor (fixed)
    parameter W_FRONT_INH  = 50,      // front sensor -> both motors (fixed, inhibitory)
    parameter W_LAT        = 80,       // motor<->motor mutual inhibition (winner-take-all between the two wheels)
    parameter W_GOAL       = 20,       // goal steering (fixed, 0 = ignore goal)
    parameter TONIC        = 30,       // motor background current per tick
    parameter NOISE_MASK   = 3,       // motor noise amplitude (0 = none)
    parameter W_INIT       = 25,       // start value of every hidden->motor weight (+-4 spread)
    parameter W_MAXV       = 120,
    parameter W_MINV       = 1,
    parameter LTP_SH       = 5,        // A+ = 2^-LTP_SH
    parameter TR_DECAY     = 24,       // trace decay per tick (time constant of the STDP window)
    parameter EL_SH        = 6,        // eligibility tag decays by 2^-EL_SH per tick
    parameter DA_SH        = 4,        // while dopamine is on, 2^-DA_SH of each tag is turned into weight change per tick
    parameter PRELOAD      = 1,        // 0 = all weights start at W_INIT, 1 = start from learned weights (use with learning off)
    parameter INIT_SPREAD  = 0,        // +- spread of the start weights (0 = all equal, choice is then made by the noise)
    parameter DECAY_PERIOD = 0,        // optional: every N ticks weights drift 1/16 back to start (0 = off)
    parameter NP           = 4 * NH    // number of plastic synapses = 2*NH hidden * 2 motors (do not change)
)(
    input              clk,
    input              reset,
    input              learn_en,
    input              da,                 // dopamine / reward
    input  [8*9-1:0]   ext_flat,           // 9 x 8-bit sensory drive
    output reg [1:0]   spike_mot,          // [0]=ML [1]=MR   (valid after tick_done)
    output reg         tick_done,
    output [16*NP-1:0] w_flat              // hidden->motor weights, 16 bit, unit = 1/16
);
    localparam NI  = 9;
    localparam NHT = 2 * NH;

        // Learned weights, one per synapse. k = 2*h + m  (m=0: ML left wheel, m=1: MR right wheel)
    // Overwrite the numbers with the ones from your last [W] lines.
    function integer learned_w; input integer kk;
        begin
            case (kk)
                // red group h0..h5
                0:  learned_w = 12;   1:  learned_w = 120;   // h0  ML, MR
                2:  learned_w = 14;   3:  learned_w = 120;   // h1
                4:  learned_w = 15;   5:  learned_w = 120;   // h2
                6:  learned_w = 15;   7:  learned_w = 120;   // h3
                8:  learned_w = 14;   9:  learned_w = 120;   // h4
                10: learned_w = 12;   11: learned_w = 120;   // h5
                // green group h6..h11
                12: learned_w = 120;  13: learned_w = 12;     // h6
                14: learned_w = 120;  15: learned_w = 12;    // h7
                16: learned_w = 120;  17: learned_w = 12;    // h8
                18: learned_w = 120;  19: learned_w = 11;     // h9
                20: learned_w = 120;  21: learned_w = 12;     // h10
                22: learned_w = 120;  23: learned_w = 12;     // h11
                default: learned_w = 25;
            endcase
        end
    endfunction

    reg [15:0] w     [0:NP-1];
    reg [15:0] w0    [0:NP-1];
    integer    el    [0:NP-1];
    reg [7:0]  tr_h  [0:NHT-1];
    reg [7:0]  tr_m  [0:1];
    reg [15:0] v_in  [0:NI-1];
    reg [15:0] v_h   [0:NHT-1];
    reg [15:0] v_m   [0:1];
    reg [NI-1:0]  sp_in;
    reg [NHT-1:0] sp_h;
    reg [15:0] lfsr;
    reg [15:0] cnt;
    reg [15:0] dcnt;

    wire tick = (cnt == TICK_DIV - 1);

    genvar g;
    generate for (g = 0; g < NP; g = g + 1) begin : WF
        assign w_flat[16*g +: 16] = w[g];
    end endgenerate

    integer i, h, m, k, grp;
    integer vn, drive, plast, t, dp, dm, wi;
    reg [NI-1:0]  sp_in_old;
    reg [NHT-1:0] sp_h_old;
    reg [1:0]     sp_m_old, sp_m_new;

    always @(posedge clk) begin
        tick_done <= 1'b0;
                     if (reset) begin
            cnt <= 16'd0; dcnt = 16'd0;
            sp_in = 0; sp_h = 0; spike_mot = 2'b00; lfsr = 16'hACE1;
            for (k = 0; k < NP; k = k + 1) begin
                if (PRELOAD != 0)
                    w0[k] = learned_w(k) * 16;      // per-neuron learned weights
                else
                    w0[k] = (W_INIT + ((((k * 37) % 9) - 4) * INIT_SPREAD) / 4) * 16;
                el[k] = 0;
                w[k]  = w0[k];
            end
            for (i = 0; i < NI;  i = i + 1) v_in[i] = 0;
            for (h = 0; h < NHT; h = h + 1) begin v_h[h] = 0; tr_h[h] = 0; end
            for (m = 0; m < 2;   m = m + 1) begin v_m[m] = 0; tr_m[m] = 0; end
        
    
        end else begin
            cnt <= tick ? 16'd0 : cnt + 16'd1;
            if (tick) begin
                tick_done <= 1'b1;
                sp_in_old = sp_in;  sp_h_old = sp_h;  sp_m_old = spike_mot;

                // ---------- traces (pre traces from hidden, post traces from motor) ----------
                for (h = 0; h < NHT; h = h + 1) begin
                    if (sp_h_old[h])          tr_h[h] = 8'd255;
                    else if (tr_h[h] > TR_DECAY) tr_h[h] = tr_h[h] - TR_DECAY;
                    else                      tr_h[h] = 8'd0;
                end

                // ---------- sensory neurons: rate code of the sensor byte ----------
                for (i = 0; i < NI; i = i + 1) begin
                    vn = v_in[i] + ext_flat[8*i +: 8];
                    if (vn >= THR_IN) begin sp_in[i] = 1'b1; v_in[i] = 0; end
                    else              begin sp_in[i] = 1'b0; v_in[i] = vn; end
                end

                // ---------- hidden neurons: group 0 hears CUE_RED, group 1 hears CUE_GREEN ----------
                for (h = 0; h < NHT; h = h + 1) begin
                    grp   = h / NH;
                    drive = sp_in_old[5 + grp] ? W_CUE : 0;
                    vn = v_h[h] + drive - LEAK_H;
                    if (vn < 0) vn = 0;
                    if (vn >= THR_H + 7 * (h % 5)) begin sp_h[h] = 1'b1; v_h[h] = 0; end
                    else                           begin sp_h[h] = 1'b0; v_h[h] = vn; end
                end

                // ---------- motor neurons ----------
                lfsr = {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
                for (m = 0; m < 2; m = m + 1) begin
                    // plastic (conditioned) input
                    plast = 0;
                    for (h = 0; h < NHT; h = h + 1)
                        if (sp_h_old[h]) begin wi = w[2*h + m]; plast = plast + (wi >> 4); end
                    if (da) plast = plast + (plast >> 3) + (plast >> 5);        // x ~1.15 with dopamine
                    // fixed (unconditioned) input
                    drive = TONIC + plast;
                    t = (lfsr >> (4*m)) & NOISE_MASK;
                    drive = drive + t;
                    if (m == 0) begin
                        if (sp_in_old[0]) drive = drive + W_REFLEX;
                        if (sp_in_old[1]) drive = drive + W_REFLEX;
                        if (sp_in_old[8]) drive = drive + W_GOAL;
                    end else begin
                        if (sp_in_old[3]) drive = drive + W_REFLEX;
                        if (sp_in_old[4]) drive = drive + W_REFLEX;
                        if (sp_in_old[7]) drive = drive + W_GOAL;
                    end
                    if (sp_in_old[2]) drive = drive - W_FRONT_INH;
                    if (m == 0 && sp_m_old[1]) drive = drive - W_LAT;
                    if (m == 1 && sp_m_old[0]) drive = drive - W_LAT;
                    vn = v_m[m] + drive - LEAK_M;
                    if (vn < 0) vn = 0;
                    if (vn >= THR_MOT) begin sp_m_new[m] = 1'b1; v_m[m] = 0; end
                    else               begin sp_m_new[m] = 1'b0; v_m[m] = vn; end
                end

                // ---------- reward-modulated STDP ----------
                if (learn_en) begin
                    // 1) dopamine turns part of every eligibility tag into a weight change
                    if (da)
                        for (k = 0; k < NP; k = k + 1) begin
                            dp = el[k] / (1 << DA_SH);
                            wi = w[k] + dp;
                            if (wi > W_MAXV * 16) wi = W_MAXV * 16;
                            if (wi < W_MINV * 16) wi = W_MINV * 16;
                            w[k] = wi;
                            el[k] = el[k] - dp;
                        end
                    // 2) tags decay
                    for (k = 0; k < NP; k = k + 1) begin
                        dp = el[k] / (1 << EL_SH);
                        if (dp == 0 && el[k] > 0) dp = 1;
                        if (dp == 0 && el[k] < 0) dp = -1;
                        el[k] = el[k] - dp;
                    end
                    // 3) LTD tag: hidden (pre) spike now, motor (post) fired earlier -> negative tag
                    for (h = 0; h < NHT; h = h + 1)
                        if (sp_h_old[h])
                            for (m = 0; m < 2; m = m + 1) begin
                                wi = w[2*h+m];
                                dm = (wi * tr_m[m]) >> (8 + LTP_SH);
                                dm = dm + (dm >> 4);
                                el[2*h+m] = el[2*h+m] - dm;
                            end
                    // 4) LTP tag: motor (post) spike now, hidden (pre) fired shortly before -> positive tag
                    for (m = 0; m < 2; m = m + 1)
                        if (sp_m_new[m])
                            for (h = 0; h < NHT; h = h + 1) begin
                                wi = w[2*h+m];
                                dp = (wi * tr_h[h]) >> (8 + LTP_SH);
                                el[2*h+m] = el[2*h+m] + dp;
                            end
                    // optional slow forgetting toward the start values
                    if (DECAY_PERIOD != 0) begin
                        dcnt = dcnt + 1;
                        if (dcnt >= DECAY_PERIOD) begin
                            dcnt = 0;
                            for (k = 0; k < NP; k = k + 1) begin
                                wi = w[k];
                                if (wi > w0[k]) w[k] = wi - 1;
                                else if (wi < w0[k]) w[k] = wi + 1;
                            end
                        end
                    end
                end

                // motor traces for the next tick (post trace)
                for (m = 0; m < 2; m = m + 1) begin
                    if (sp_m_new[m])             tr_m[m] = 8'd255;
                    else if (tr_m[m] > TR_DECAY) tr_m[m] = tr_m[m] - TR_DECAY;
                    else                         tr_m[m] = 8'd0;
                end
                spike_mot = sp_m_new;
            end
        end
    end
endmodule