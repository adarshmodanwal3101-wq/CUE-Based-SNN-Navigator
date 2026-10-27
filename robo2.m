% =========================================================================
%  SNN CO-SIMULATION ROBOT - cue-conditioned controller (after Xue et al. 2014)
%
%  Sends 10 bytes per step:
%     [LeftOuter LeftInner Front RightInner RightOuter CueRed CueGreen GoalLeft GoalRight Dopamine]
%  Receives 1 byte:   bits[3:0] = LEFT-wheel spike count,  bits[7:4] = RIGHT-wheel spike count
%
%  Start order:  1) vvp -M. -mbridge snn_sim.vvp | tee run.log   (waits for MATLAB)
%                2) run this script
%  Weights are printed in the vvp window ([W] lines): rows ML, MR; columns = 6 red-group
%  hidden neurons, then 6 green-group hidden neurons.
%
%  Ideas taken from the paper:
%   - fixed obstacle-avoidance reflex + plastic cue -> motor pathway
%   - two motor neurons = two wheels, spike counts set wheel speeds
%   - equal counts at an obstacle -> random rotation
%   - dopamine (reward) is sent shortly AFTER a correct turn at a signpost
% =========================================================================
clc; clear; close all;

% ============================== CONFIG ===================================
cfg.dt           = 0.05;
cfg.tMax         = 800;
cfg.startPose    = [1; 1; pi/2];
cfg.goalPos      = [9; 1];
cfg.goalRadius   = 0.45;
cfg.robotRadius  = 0.2;
cfg.maxRange     = 4.0;
cfg.sensorAngles = [pi/3, pi/6, 0, -pi/6, -pi/3];   % [LO LI F RI RO]

cfg.vFwd         = 1;          % m/s when both wheels reach cref spikes
cfg.wTurn        = 0.9;        % rad/s at full wheel imbalance
cfg.cref         = 8;          % spike count that means "full" for one wheel
cfg.goalDeadzone = deg2rad(5);
cfg.goalFullScale= pi/2;

% signposts: [x  y  colour(1=red,2=green)  correct turn(+1 left, -1 right)]
% The network is NOT told what a colour means: it must learn red -> left, green -> right.
cfg.signs = [ 1.8 6.2 2 -1;
              3.8 8.3 2 -1;
              3.5 2.5 1 +1;
              5.5 1.5 1 +1;
              6.8 6.9 2 -1;
              8.3 7.0 2 -1 ];
cfg.cueRange     = 1.6;        % colour sensors see a signpost closer than this (in front only)
cfg.teachRange   = 1.3;        % dopamine can be given inside this distance
cfg.tieDist      = 0.6;        % equal counts + obstacle closer than this -> random rotation
cfg.escapeSteps  = 3;          % consecutive collisions before an escape turn
cfg.escapeHold   = 60;         % length of the escape turn (steps)
cfg.verbose      = false;

cfg.host = '127.0.0.1';  cfg.port = 5000;  cfg.timeoutSec = 20;
% =========================================================================

dt = cfg.dt;  time = 0:dt:cfg.tMax;
robotPose = cfg.startPose;  goalPos = cfg.goalPos;
numSensors = numel(cfg.sensorAngles);

% ---------------- Environment map: [x_min, y_min, width, height] ----------
mapBlocks = [
    0, 0, 10, 0.2;   0, 9.8, 10, 0.2;   0, 0, 0.2, 10;   9.8, 0, 0.2, 10;
    2.5, 0, 0.2, 7;  5, 3, 0.2, 6.8;    7.5, 0, 0.2, 7;
    0.2, 3.2, 0.9, 0.5;  1.6, 5.2, 0.9, 0.5;  1, 8.2, 0.6, 0.6;  2.7, 6, 0.9, 0.5;
    4.1, 4.2, 0.9, 0.5;  3.2, 1.2, 0.6, 0.6;  6.6, 1.8, 0.9, 0.5; 5.2, 4, 0.9, 0.5;
    6.6, 5.5, 0.9, 0.5;  6, 8.2, 0.6, 0.6;    8.9, 7.4, 0.7, 0.5 ];
mapWalls = block_to_walls(mapBlocks);

% ============================ GRAPHICS ===================================
fig = figure('Name', 'SNN Cue-Conditioned Robot', 'Color', [0.08 0.09 0.11], 'Position', [100 100 1200 650]);
ax_map = subplot('Position', [0.05 0.08 0.58 0.85]);
hold(ax_map, 'on'); grid(ax_map, 'on'); axis(ax_map, 'equal'); axis(ax_map, [-0.5 10.5 -0.5 10.5]);
set(ax_map, 'Color', [0.12 0.13 0.16], 'XColor', [0.5 0.5 0.6], 'YColor', [0.5 0.5 0.6], ...
    'GridColor', [0.2 0.25 0.3], 'GridAlpha', 0.5);
title(ax_map, '2D ROBOT ARENA', 'Color', [0.9 0.9 0.95], 'FontSize', 12);
ax_hud = subplot('Position', [0.68 0.08 0.28 0.85]);
axis(ax_hud, [0 10 0 10]); axis(ax_hud, 'off');

for b = 1:size(mapBlocks, 1)
    rectangle('Position', mapBlocks(b,:), 'Parent', ax_map, 'FaceColor', [0.25 0.28 0.35], ...
        'EdgeColor', [0.4 0.45 0.55], 'LineWidth', 1.5);
end
circle_patch(ax_map, goalPos(1), goalPos(2), 0.4, [0.1 0.8 0.4], 0.2);
plot(ax_map, goalPos(1), goalPos(2), 'gp', 'MarkerSize', 14, 'MarkerFaceColor', [0.2 1 0.4], 'MarkerEdgeColor', 'w');
for k = 1:size(cfg.signs, 1)                       % draw signposts
    col = [0.9 0.2 0.2]; if cfg.signs(k,3) == 2, col = [0.2 0.9 0.3]; end
    circle_patch(ax_map, cfg.signs(k,1), cfg.signs(k,2), 0.15, col, 0.9);
end
h_trail = plot(ax_map, nan, nan, 'Color', [0 0.8 1], 'LineWidth', 2);
path_x = []; path_y = [];

sensorNames = {'LEFT  OUTER ', 'LEFT  INNER ', 'FRONT       ', 'RIGHT INNER ', 'RIGHT OUTER '};
h_sensor_bars = zeros(1, numSensors);
for s = 1:numSensors
    yb = 8.7 - s*0.9;
    text(0.5, yb + 0.22, sensorNames{s}, 'Parent', ax_hud, 'Color', [0.7 0.7 0.8], 'FontName', 'Monospaced', 'FontSize', 9);
    rectangle('Position', [3.8, yb, 5.5, 0.45], 'Parent', ax_hud, 'FaceColor', [0.18 0.2 0.25], 'EdgeColor', [0.3 0.3 0.35]);
    h_sensor_bars(s) = rectangle('Position', [3.8, yb, 5.5, 0.45], 'Parent', ax_hud, 'FaceColor', [0 0.8 0.4], 'EdgeColor', 'none');
end
mk = @(y, str, col) text(0.5, y, str, 'Parent', ax_hud, 'Color', col, 'FontName', 'Monospaced', 'FontSize', 10, 'FontWeight', 'bold');
h_c_text = mk(4.1, 'Cue / dopamine : -', [1.0 0.5 0.9]);
h_m_text = mk(3.4, 'Wheels L / R   : - / -', [1.0 0.6 0.2]);
h_v_text = mk(2.7, 'Linear Speed   : 0.00 m/s', [0.0 0.8 1.0]);
h_w_text = mk(2.0, 'Angular Speed  : 0.00 rad/s', [0.0 0.8 1.0]);
h_g_text = mk(1.3, 'Goal Distance  : 0.00 m', [1.0 0.8 0.2]);
h_s_text = mk(0.6, 'Status         : -', [0.8 0.8 0.8]);
drawnow;

% ============================ TCP CONNECT =================================
snn_brain = tcpclient(cfg.host, cfg.port, 'Timeout', cfg.timeoutSec);

% ============================ SIMULATION ==================================
collided = false;  nCollisions = 0;
pendDA = 0;                 % dopamine to send on the NEXT step (reward follows the action)
tieDir = 0;  idleCount = 0;  escapeLeft = 0;  escapeDir = 1;
zoneTurns = 0;  zoneCorrect = 0;
h_body = []; h_dir = []; h_rays = [];

try
for step = 1:length(time)
    if ~isvalid(fig), break; end
    currX = robotPose(1); currY = robotPose(2); currTheta = robotPose(3);
    path_x(end+1) = currX; path_y(end+1) = currY; %#ok<SAGROW>

    distToGoal = norm(goalPos - [currX; currY]);
    if distToGoal < cfg.goalRadius
        title(ax_map, 'GOAL REACHED!', 'Color', [0.2 1 0.4], 'FontSize', 14); break;
    end

    % --- A: distance sensors ---
    sd = zeros(1, numSensors);
    for s = 1:numSensors
        sd(s) = cast_ray([currX, currY], currTheta + cfg.sensorAngles(s), cfg.maxRange, mapWalls);
    end
    dFront = sd(3);  dLeft = min(sd([1 2]));  dRight = min(sd([4 5]));
    snn_inputs = uint8(round((1 - sd / cfg.maxRange) * 255));

    % --- B: goal direction (positive error = goal on the LEFT) ---
    bearing = atan2(goalPos(2) - currY, goalPos(1) - currX);
    err = angdiff(currTheta, bearing);
    mag = min(1, max(0, abs(err) - cfg.goalDeadzone) / cfg.goalFullScale);
    goalL = uint8(round(255 * mag * (err > 0)));
    goalR = uint8(round(255 * mag * (err < 0)));

    % --- C: colour sensors: nearest signpost in front of the robot ---
    cueRed = uint8(0); cueGreen = uint8(0); nearD = inf; nearTurn = 0;
    for k = 1:size(cfg.signs, 1)
        dd = hypot(cfg.signs(k,1) - currX, cfg.signs(k,2) - currY);
        bb = angdiff(currTheta, atan2(cfg.signs(k,2) - currY, cfg.signs(k,1) - currX));
        if dd < cfg.cueRange && abs(bb) < deg2rad(100) && dd < nearD
            nearD = dd;  nearTurn = cfg.signs(k,4);
            inten = uint8(round(255 * (1 - dd / cfg.cueRange)));
            if cfg.signs(k,3) == 1, cueRed = inten; cueGreen = 0; else, cueGreen = inten; cueRed = 0; end
        end
    end

    % --- D: exchange with Verilog (10 bytes out, 1 byte back) ---
    write(snn_brain, [snn_inputs, cueRed, cueGreen, goalL, goalR, uint8(pendDA)], 'uint8');
    spike_byte = read(snn_brain, 1, 'uint8');
    if isempty(spike_byte)
        warning('No reply from the simulator (timeout or it closed). Stopping.'); break;
    end
    cL = double(bitand(spike_byte, 15));
    cR = double(bitshift(spike_byte, -4));

    % --- E: wheels -> motion (right wheel faster = turn LEFT) ---
    v_cmd = cfg.vFwd  * min(1, (cL + cR) / (2*cfg.cref));
    w_cmd = cfg.wTurn * max(-1, min(1, (cR - cL) / cfg.cref));
    status = 'ok';

    if nearD < cfg.teachRange && abs(w_cmd) > 0.25*cfg.wTurn
        ok = (sign(w_cmd) == nearTurn);
        pendDA = double(ok);
        if strcmp(status,'ok') && min([dFront dLeft dRight]) > 1.0   % cue-driven, no wall nearby
            zoneTurns = zoneTurns + 1;  zoneCorrect = zoneCorrect + ok;
        end
    end

    % stuck detector: repeated collisions -> escape turn, direction chosen ONCE and held
  if collided || (cL + cR == 0), idleCount = idleCount + 1; else, idleCount = 0; end  
    if idleCount >= cfg.escapeSteps && escapeLeft == 0
        escapeLeft = cfg.escapeHold;
        escapeDir  = 2*(dLeft > dRight) - 1;
    end
    if escapeLeft > 0
        escapeLeft = escapeLeft - 1;  v_cmd = 0;  w_cmd = cfg.wTurn * escapeDir;  status = 'ESCAPE turn';
    end

    % teacher: dopamine for the NEXT step if the robot is turning the correct way at a signpost
    pendDA = 0;
    if nearD < cfg.teachRange && abs(w_cmd) > 0.25*cfg.wTurn
        ok = (sign(w_cmd) == nearTurn);
        pendDA = double(ok);  zoneTurns = zoneTurns + 1;  zoneCorrect = zoneCorrect + ok;
    end

    % --- F: update pose with collision check ---
    newPos = [currX; currY] + v_cmd * [cos(currTheta); sin(currTheta)] * dt;
    collided = false;
    if v_cmd > 0 && hits_wall(newPos, mapWalls, cfg.robotRadius)
        collided = true; nCollisions = nCollisions + 1; newPos = [currX; currY]; v_cmd = 0; status = 'COLLISION';
    end
    robotPose(1:2) = newPos;
    robotPose(3)   = angdiff(0, currTheta + w_cmd * dt);

    if cfg.verbose
        fprintf('step %4d | cue R/G=%3d/%3d | L/R wheels=%2d/%2d | DA=%d | %s\n', step, cueRed, cueGreen, cL, cR, pendDA, status);
    end

    % --- G: render ---
    set(h_trail, 'XData', path_x, 'YData', path_y);
    if ~isempty(h_body) && isvalid(h_body), delete(h_body); end
    if ~isempty(h_dir)  && isvalid(h_dir),  delete(h_dir);  end
    delete(h_rays(isgraphics(h_rays)));
    h_rays = gobjects(1, numSensors);
    for s = 1:numSensors
        rayA = currTheta + cfg.sensorAngles(s);  rLen = sd(s);
        endP = [currX + rLen*cos(rayA), currY + rLen*sin(rayA)];  fr = rLen / cfg.maxRange;
        h_rays(s) = plot(ax_map, [currX, endP(1)], [currY, endP(2)], 'Color', [1-fr, fr, 0.2], 'LineWidth', 2);
    end
    h_body = circle_patch(ax_map, currX, currY, cfg.robotRadius, [0 0.6 1], 0.8);
    h_dir  = plot(ax_map, [currX, currX + 0.5*cos(currTheta)], [currY, currY + 0.5*sin(currTheta)], 'w-', 'LineWidth', 2.5);
    for s = 1:numSensors
        yb = 8.7 - s*0.9;  fr = sd(s) / cfg.maxRange;
        set(h_sensor_bars(s), 'Position', [3.8, yb, max(0.01, 5.5*fr), 0.45], 'FaceColor', [1-fr, fr, 0.2]);
    end
    set(h_c_text, 'String', sprintf('Cue R/G %3d/%3d  DA %d', cueRed, cueGreen, pendDA));
    set(h_m_text, 'String', sprintf('Wheels L / R   : %d / %d', cL, cR));
    set(h_v_text, 'String', sprintf('Linear Speed   : %.2f m/s', v_cmd));
    set(h_w_text, 'String', sprintf('Angular Speed  : %.2f rad/s', w_cmd));
    set(h_g_text, 'String', sprintf('Goal Distance  : %.2f m  coll %d', distToGoal, nCollisions));
    set(h_s_text, 'String', ['Status         : ' status]);
    drawnow limitrate;
end
catch ME
    warning('Simulation stopped: %s', ME.message);
end
fprintf('RESULT: steps=%d  collisions=%d  goalDist=%.2f  signpost turns=%d correct=%d (%.0f%%)\n', ...
    step, nCollisions, distToGoal, zoneTurns, zoneCorrect, 100*zoneCorrect/max(1,zoneTurns));
clear snn_brain;

% ========================== GEOMETRY HELPERS ==============================
function h = circle_patch(ax, x, y, r, color, alpha)
    th = 0:0.1:2*pi;
    h = patch('XData', r*cos(th) + x, 'YData', r*sin(th) + y, 'Parent', ax, 'FaceColor', color, ...
        'EdgeColor', 'w', 'FaceAlpha', alpha, 'LineWidth', 1);
end
function walls = block_to_walls(blocks)
    walls = [];
    for b = 1:size(blocks, 1)
        x = blocks(b,1); y = blocks(b,2); w = blocks(b,3); h = blocks(b,4);
        walls = [walls; x, y, x+w, y; x+w, y, x+w, y+h; x+w, y+h, x, y+h; x, y+h, x, y]; %#ok<AGROW>
    end
end
function d = angdiff(th1, th2)
    d = mod(th2 - th1 + pi, 2*pi) - pi;
end
function tf = hits_wall(pos, walls, radius)
    tf = false; p = pos(:)';
    for w = 1:size(walls, 1)
        a = walls(w, 1:2); b = walls(w, 3:4); ab = b - a;
        t = max(0, min(1, dot(p - a, ab) / dot(ab, ab)));
        if norm(p - (a + t*ab)) < radius, tf = true; return; end
    end
end
function minDist = cast_ray(origin, angle, maxR, walls)
    minDist = maxR; p1 = origin; p2 = origin + [maxR * cos(angle), maxR * sin(angle)];
    for w = 1:size(walls, 1)
        q1 = walls(w, 1:2); q2 = walls(w, 3:4); r = p2 - p1; s = q2 - q1;
        rxs = r(1)*s(2) - r(2)*s(1);
        if rxs ~= 0
            q_p = q1 - p1;
            t = (q_p(1)*s(2) - q_p(2)*s(1)) / rxs;
            u = (q_p(1)*r(2) - q_p(2)*r(1)) / rxs;
            if t >= 0 && t <= 1 && u >= 0 && u <= 1
                dist = t * maxR;  if dist < minDist, minDist = dist; end
            end
        end
    end
end