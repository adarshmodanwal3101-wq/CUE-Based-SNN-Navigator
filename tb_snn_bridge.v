// ============================================================================
//  tb_snn_paper.v - co-simulation testbench for snn_paper.v  (MATLAB <-> Verilog over TCP)
//
//  MATLAB -> Verilog : 10 bytes
//     [LeftOuter, LeftInner, Front, RightInner, RightOuter, CueRed, CueGreen, GoalLeft, GoalRight, Dopamine]
//  Verilog -> MATLAB : 1 byte    bits[3:0] = left-wheel spike count (0..15)
//                                bits[7:4] = right-wheel spike count (0..15)
//
//  Build: iverilog -I. -o snn_sim.vvp tb_snn_paper.v snn_paper.v
//  Run  : vvp -M. -mbridge snn_sim.vvp | tee run.log
// ============================================================================
`timescale 1ns/1ps
module tb_snn_paper;
    localparam LEARN_ENABLE   = 1;     // 0 = frozen weights, 1 = STDP + dopamine
    localparam NH             = 6;
    localparam NP             = 4*NH;
    localparam TICK_DIV       = 4;
    localparam TICKS_PER_STEP = 32;    // neuron ticks per MATLAB step (spike-count window)
    localparam PRINT_EVERY    = 20;

    localparam [7:0] NEAR_START = 8'd120;   // obstacle ignored below this (~1 m away)
    localparam       NEAR_SHIFT = 1;
    localparam       GOAL_SHIFT = 2;
    localparam       CUE_SHIFT  = 1;

    function [7:0] enc_near; input [7:0] x;
        begin enc_near = (x > NEAR_START) ? ((x - NEAR_START) >> NEAR_SHIFT) : 8'd0; end
    endfunction

    reg clk = 0, reset = 1, da = 0;
    wire learn_en = (LEARN_ENABLE != 0);
    reg  [71:0]  ext = 72'd0;
    wire [1:0]   spk;
    wire         tick_done;
    wire [16*NP-1:0] w_flat;

    snn_paper #(.NH(NH), .TICK_DIV(TICK_DIV),.PRELOAD(0)) dut (
        .clk(clk), .reset(reset), .learn_en(learn_en), .da(da),
        .ext_flat(ext), .spike_mot(spk), .tick_done(tick_done), .w_flat(w_flat));

    always #5 clk = ~clk;

    reg ok, connected;
    integer b_lo, b_li, b_f, b_ri, b_ro, b_cr, b_cg, b_gl, b_gr, b_da;
    integer c_l, c_r, t, step, h;
    reg [7:0] motor_byte;

    initial begin
        connected = 1; step = 0;
        $display("[Verilog] snn_paper: 9 sensory, %0d hidden, 2 motor, learn=%0d", 2*NH, LEARN_ENABLE);
        $bridge_open();
        #40 reset = 0;
        while (connected) begin
            $bridge_read(ok, b_lo, b_li, b_f, b_ri, b_ro, b_cr, b_cg, b_gl, b_gr, b_da);
            if (!ok) begin
                $display("[Verilog] MATLAB disconnected."); connected = 0;
            end else begin
                @(negedge clk);
                ext[8*0 +: 8] = enc_near(b_lo[7:0]);
                ext[8*1 +: 8] = enc_near(b_li[7:0]);
                ext[8*2 +: 8] = enc_near(b_f [7:0]);
                ext[8*3 +: 8] = enc_near(b_ri[7:0]);
                ext[8*4 +: 8] = enc_near(b_ro[7:0]);
                ext[8*5 +: 8] = b_cr[7:0] >> CUE_SHIFT;
                ext[8*6 +: 8] = b_cg[7:0] >> CUE_SHIFT;
                ext[8*7 +: 8] = b_gl[7:0] >> GOAL_SHIFT;
                ext[8*8 +: 8] = b_gr[7:0] >> GOAL_SHIFT;
                da = b_da[0];                       // dopamine held for the whole window

                c_l = 0; c_r = 0;
                for (t = 0; t < TICKS_PER_STEP*TICK_DIV; t = t + 1) begin
                    @(negedge clk);
                    if (tick_done) begin c_l = c_l + spk[0]; c_r = c_r + spk[1]; end
                end
                if (c_l > 15) c_l = 15;
                if (c_r > 15) c_r = 15;
                motor_byte = {c_r[3:0], c_l[3:0]};

                $display("[RX] LO=%0d LI=%0d F=%0d RI=%0d RO=%0d CR=%0d CG=%0d GL=%0d GR=%0d DA=%0d -> L/R=%0d/%0d",
                         b_lo[7:0], b_li[7:0], b_f[7:0], b_ri[7:0], b_ro[7:0], b_cr[7:0], b_cg[7:0],
                         b_gl[7:0], b_gr[7:0], b_da[0], c_l, c_r);
                $bridge_write(motor_byte);

                step = step + 1;
                if (step % PRINT_EVERY == 0) begin
                    // weights in real units (stored x16). Row ML then MR. Columns: H1 (red group) then H2 (green group)
                    $write("[W] ML:");
                    for (h = 0; h < 2*NH; h = h + 1) $write(" %0d", w_flat[16*(2*h)   +: 16] >> 4);
                    $display("");
                    $write("[W] MR:");
                    for (h = 0; h < 2*NH; h = h + 1) $write(" %0d", w_flat[16*(2*h+1) +: 16] >> 4);
                    $display("");
                end
            end
        end
        $bridge_close();
        $finish;
    end
endmodule