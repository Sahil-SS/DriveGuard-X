%% ============================================================
% DRIVEGUARD-X BATTERY AI PREDICTION
% 13S Lithium-Ion Battery
%
% Simulink
%    -> Logged Battery Voltage / Current / Temperature
%    -> 13S Cell-Equivalent Conversion
%    -> 21 Battery Features
%    -> FastAPI
%    -> Random Forest
%    -> Battery SOH
%% ============================================================

clc;

fprintf('\n');
fprintf('============================================\n');
fprintf('       DRIVEGUARD-X BATTERY AI              \n');
fprintf('============================================\n');


%% ============================================================
% 1. READ LOGGED SIMULINK DATA
%% ============================================================

fprintf('\n[1/7] Reading Simulink data...\n');

% Battery Current
I = out.logsout{1}.Values;

% Battery Temperature
T = out.logsout{2}.Values;

% Battery Voltage
V = out.logsout{3}.Values;

% Extract time and data
time = I.Time;
current_pack = I.Data;
temperature = T.Data;
voltage_pack = V.Data;

fprintf('Battery Current samples: %d\n', length(current_pack));
fprintf('Battery Temperature samples: %d\n', length(temperature));
fprintf('Battery Voltage samples: %d\n', length(voltage_pack));


%% ============================================================
% 2. 13S LITHIUM-ION BATTERY CONVERSION
%% ============================================================

fprintf('\n[2/7] Applying 13S battery conversion...\n');

% Your simulated EV battery is treated as a 13S Li-ion pack.
SERIES_CELLS = 13;

% Convert 48 V-class pack voltage into cell-equivalent voltage.
voltage_measured = voltage_pack / SERIES_CELLS;

% Your Simulink battery current is positive during discharge.
% NASA training data represents discharge current as negative.
current_measured = -abs(current_pack);

% Load current magnitude.
current_load = abs(current_pack);

% There is currently no separate load-voltage sensor.
% Therefore, use the battery terminal voltage as load voltage.
voltage_load = voltage_measured;

fprintf('Number of series cells: %d\n', SERIES_CELLS);
fprintf('Initial pack voltage: %.4f V\n', voltage_pack(1));
fprintf('Initial cell-equivalent voltage: %.4f V\n', ...
    voltage_measured(1));


%% ============================================================
% 3. REMOVE INVALID DATA
%% ============================================================

fprintf('\n[3/7] Checking data validity...\n');

valid = ...
    ~isnan(time) & ~isinf(time) & ...
    ~isnan(voltage_measured) & ~isinf(voltage_measured) & ...
    ~isnan(current_measured) & ~isinf(current_measured) & ...
    ~isnan(temperature) & ~isinf(temperature) & ...
    ~isnan(current_load) & ~isinf(current_load) & ...
    ~isnan(voltage_load) & ~isinf(voltage_load);

% Apply validity mask
time = time(valid);
voltage_measured = voltage_measured(valid);
current_measured = current_measured(valid);
temperature = temperature(valid);
current_load = current_load(valid);
voltage_load = voltage_load(valid);

fprintf('Valid samples remaining: %d\n', length(time));


%% ============================================================
% 4. IDENTIFY ACTIVE DISCHARGE REGION
%% ============================================================

fprintf('\n[4/7] Detecting active discharge region...\n');

% Final notebook logic:
% Active load region is identified using load current magnitude > 0.1 A.
active_mask = abs(current_load) > 0.1;

active_idx = find(active_mask);


% Fallback 1:
% Use measured discharge current.
if length(active_idx) < 5

    fprintf('Less than 5 active samples found.\n');
    fprintf('Using measured current as fallback...\n');

    active_idx = find(abs(current_measured) > 0.1);

end


% Fallback 2:
% If still insufficient, use the complete simulation.
if length(active_idx) < 5

    fprintf('Still less than 5 active samples found.\n');
    fprintf('Using complete simulation data...\n');

    active_idx = (1:length(time))';

end

fprintf('Active samples used: %d\n', length(active_idx));


%% ============================================================
% 5. EXTRACT ACTIVE DATA
%% ============================================================

fprintf('\n[5/7] Preparing active battery data...\n');

v = voltage_measured(active_idx);
i = current_measured(active_idx);
temp = temperature(active_idx);

i_load = abs(current_load(active_idx));
v_load = voltage_load(active_idx);

t = time(active_idx);


%% ============================================================
% 6. CALCULATE FINAL 21 BATTERY FEATURES
%% ============================================================

fprintf('\n[6/7] Calculating 21 AI features...\n');


% ------------------------------------------------------------
% Voltage features
% ------------------------------------------------------------

features.voltage_initial = v(1);

features.voltage_final = v(end);

features.voltage_min = min(v);

features.voltage_max = max(v);

features.voltage_mean = mean(v);

% Population standard deviation.
features.voltage_std = std(v, 1);

features.voltage_drop = ...
    features.voltage_initial - features.voltage_final;


% ------------------------------------------------------------
% Current features
% ------------------------------------------------------------

features.current_mean = mean(i);

features.current_min = min(i);

features.current_max = max(i);

% Population standard deviation.
features.current_std = std(i, 1);


% ------------------------------------------------------------
% Temperature features
% ------------------------------------------------------------

features.temperature_initial = temp(1);

features.temperature_final = temp(end);

features.temperature_max = max(temp);

features.temperature_mean = mean(temp);

features.temperature_rise = ...
    features.temperature_final - ...
    features.temperature_initial;


% ------------------------------------------------------------
% Load current features
% ------------------------------------------------------------

features.load_current_abs_mean = mean(i_load);

features.load_current_abs_max = max(i_load);


% ------------------------------------------------------------
% Load voltage features
% ------------------------------------------------------------

features.load_voltage_mean = mean(v_load);

features.load_voltage_min = min(v_load);


% ------------------------------------------------------------
% Discharge duration
% ------------------------------------------------------------

features.discharge_duration = ...
    max(0, t(end) - t(1));


%% ============================================================
% 7. DISPLAY THE 21 FEATURES
%% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('       21 BATTERY AI FEATURES               \n');
fprintf('============================================\n');

fprintf(' 1. voltage_initial          = %.6f\n', ...
    features.voltage_initial);

fprintf(' 2. voltage_final            = %.6f\n', ...
    features.voltage_final);

fprintf(' 3. voltage_min              = %.6f\n', ...
    features.voltage_min);

fprintf(' 4. voltage_max              = %.6f\n', ...
    features.voltage_max);

fprintf(' 5. voltage_mean             = %.6f\n', ...
    features.voltage_mean);

fprintf(' 6. voltage_std              = %.6f\n', ...
    features.voltage_std);

fprintf(' 7. voltage_drop             = %.6f\n', ...
    features.voltage_drop);

fprintf(' 8. current_mean             = %.6f\n', ...
    features.current_mean);

fprintf(' 9. current_min              = %.6f\n', ...
    features.current_min);

fprintf('10. current_max              = %.6f\n', ...
    features.current_max);

fprintf('11. current_std              = %.6f\n', ...
    features.current_std);

fprintf('12. temperature_initial      = %.6f\n', ...
    features.temperature_initial);

fprintf('13. temperature_final        = %.6f\n', ...
    features.temperature_final);

fprintf('14. temperature_max          = %.6f\n', ...
    features.temperature_max);

fprintf('15. temperature_mean         = %.6f\n', ...
    features.temperature_mean);

fprintf('16. temperature_rise         = %.6f\n', ...
    features.temperature_rise);

fprintf('17. load_current_abs_mean    = %.6f\n', ...
    features.load_current_abs_mean);

fprintf('18. load_current_abs_max     = %.6f\n', ...
    features.load_current_abs_max);

fprintf('19. load_voltage_mean        = %.6f\n', ...
    features.load_voltage_mean);

fprintf('20. load_voltage_min         = %.6f\n', ...
    features.load_voltage_min);

fprintf('21. discharge_duration       = %.6f\n', ...
    features.discharge_duration);

fprintf('============================================\n');


%% ============================================================
% 8. SAVE FEATURES
%% ============================================================

fprintf('\nSaving calculated features...\n');

save('battery_ai_features.mat', 'features');

fprintf('Saved as: battery_ai_features.mat\n');


%% ============================================================
% 9. SEND FEATURES TO FASTAPI
%% ============================================================

fprintf('\nSending features to FastAPI...\n');

% FastAPI address
url = 'http://127.0.0.1:8000/api/v1/battery/predict';

% HTTP options
options = weboptions( ...
    'MediaType', 'application/json', ...
    'Timeout', 30);

% Send the 21-feature structure to FastAPI
response = webwrite( ...
    url, ...
    features, ...
    options);


%% ============================================================
% 10. DISPLAY AI RESULT
%% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('       DRIVEGUARD-X BATTERY AI RESULT       \n');
fprintf('============================================\n');

fprintf('Model: %s\n', response.model);

fprintf('SOH: %.6f\n', response.soh);

fprintf('SOH: %.2f %%\n', response.soh_percent);

fprintf('Input features: %d\n', ...
    response.input_feature_count);

fprintf('============================================\n');

fprintf('\nBattery AI prediction completed successfully.\n');
fprintf('\n');
