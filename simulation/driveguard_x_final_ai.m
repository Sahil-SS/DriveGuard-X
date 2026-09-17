%% ============================================================
% DRIVEGUARD-X FINAL AI INTEGRATION
%
% SIMULINK
%    |
%    +--> Battery Voltage / Current / Temperature
%    |        |
%    |     21 Features
%    |        |
%    |     FastAPI
%    |        |
%    |     Battery SOH
%    |
%    +--> IMU AccX/Y/Z + GyroX/Y/Z
%             |
%          4-sample window
%             |
%           FastAPI
%             |
%       Driving Behaviour
%
% FINAL COMMAND WINDOW DEMO
%% ============================================================

clc;

fprintf('\n');
fprintf('============================================================\n');
fprintf('              DRIVEGUARD-X FINAL AI DEMO                   \n');
fprintf('============================================================\n');


%% ============================================================
% 1. CHECK SIMULINK OUTPUT
%% ============================================================

fprintf('\nChecking Simulink output...\n');

if ~exist('out','var')
    error('Simulation output "out" does not exist. Run Simulink first.');
end


%% ============================================================
% 2. BATTERY AI
%% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    BATTERY AI                              \n');
fprintf('============================================================\n');


%% Read logged battery signals

fprintf('\n[1/7] Reading Simulink battery data...\n');

I = out.logsout{1}.Values;
T = out.logsout{2}.Values;
V = out.logsout{3}.Values;

time = I.Time;

current_pack = I.Data;
temperature = T.Data;
voltage_pack = V.Data;

fprintf('Battery Current samples: %d\n', length(current_pack));
fprintf('Battery Temperature samples: %d\n', length(temperature));
fprintf('Battery Voltage samples: %d\n', length(voltage_pack));


%% 13S battery conversion

fprintf('\n[2/7] Applying 13S battery conversion...\n');

SERIES_CELLS = 13;

voltage_measured = voltage_pack / SERIES_CELLS;

current_measured = -abs(current_pack);

current_load = abs(current_pack);

voltage_load = voltage_measured;


fprintf('Number of series cells: %d\n', SERIES_CELLS);
fprintf('Initial pack voltage: %.4f V\n', voltage_pack(1));
fprintf('Initial cell-equivalent voltage: %.4f V\n', ...
    voltage_measured(1));


%% Remove invalid data

fprintf('\n[3/7] Checking data validity...\n');

valid = ...
    ~isnan(time) & ~isinf(time) & ...
    ~isnan(voltage_measured) & ~isinf(voltage_measured) & ...
    ~isnan(current_measured) & ~isinf(current_measured) & ...
    ~isnan(temperature) & ~isinf(temperature) & ...
    ~isnan(current_load) & ~isinf(current_load) & ...
    ~isnan(voltage_load) & ~isinf(voltage_load);


time = time(valid);
voltage_measured = voltage_measured(valid);
current_measured = current_measured(valid);
temperature = temperature(valid);
current_load = current_load(valid);
voltage_load = voltage_load(valid);

fprintf('Valid samples remaining: %d\n', length(time));


%% Identify active discharge region

fprintf('\n[4/7] Detecting active discharge region...\n');

active_mask = abs(current_load) > 0.1;

active_idx = find(active_mask);


if length(active_idx) < 5

    fprintf('Less than 5 active samples found.\n');
    fprintf('Using measured current as fallback...\n');

    active_idx = find(abs(current_measured) > 0.1);

end


if length(active_idx) < 5

    fprintf('Still less than 5 active samples found.\n');
    fprintf('Using complete simulation data...\n');

    active_idx = (1:length(time))';

end


fprintf('Active samples used: %d\n', length(active_idx));


%% Extract active battery data

fprintf('\n[5/7] Preparing active battery data...\n');

v = voltage_measured(active_idx);
i = current_measured(active_idx);
temp = temperature(active_idx);

i_load = abs(current_load(active_idx));
v_load = voltage_load(active_idx);

t = time(active_idx);


%% Calculate 21 battery features

fprintf('\n[6/7] Calculating 21 AI features...\n');


% Voltage features

features.voltage_initial = v(1);

features.voltage_final = v(end);

features.voltage_min = min(v);

features.voltage_max = max(v);

features.voltage_mean = mean(v);

features.voltage_std = std(v, 1);

features.voltage_drop = ...
    features.voltage_initial - features.voltage_final;


% Current features

features.current_mean = mean(i);

features.current_min = min(i);

features.current_max = max(i);

features.current_std = std(i, 1);


% Temperature features

features.temperature_initial = temp(1);

features.temperature_final = temp(end);

features.temperature_max = max(temp);

features.temperature_mean = mean(temp);

features.temperature_rise = ...
    features.temperature_final - ...
    features.temperature_initial;


% Load current features

features.load_current_abs_mean = mean(i_load);

features.load_current_abs_max = max(i_load);


% Load voltage features

features.load_voltage_mean = mean(v_load);

features.load_voltage_min = min(v_load);


% Discharge duration

features.discharge_duration = ...
    max(0, t(end) - t(1));


%% Display features

fprintf('\n');
fprintf('--------------------------------------------\n');
fprintf('       21 BATTERY AI FEATURES\n');
fprintf('--------------------------------------------\n');

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

fprintf('--------------------------------------------\n');


%% Save battery features

save('battery_ai_features.mat', 'features');

fprintf('\nBattery features saved.\n');


%% Send battery features to FastAPI

fprintf('\nSending battery features to FastAPI...\n');

battery_url = ...
    'http://127.0.0.1:8000/api/v1/battery/predict';

options = weboptions( ...
    'MediaType', 'application/json', ...
    'Timeout', 30);


battery_response = webwrite( ...
    battery_url, ...
    features, ...
    options);


%% Display battery result

fprintf('\n');
fprintf('--------------------------------------------\n');
fprintf('       BATTERY AI RESULT\n');
fprintf('--------------------------------------------\n');

fprintf('Model: %s\n', ...
    battery_response.model);

fprintf('SOH: %.6f\n', ...
    battery_response.soh);

fprintf('SOH: %.2f %%\n', ...
    battery_response.soh_percent);

fprintf('Input features: %d\n', ...
    battery_response.input_feature_count);

fprintf('--------------------------------------------\n');


%% ============================================================
% 3. IMU AI
%% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                     IMU AI                                 \n');
fprintf('============================================================\n');


%% Check IMU data

fprintf('\n[IMU] Reading Simulink IMU data...\n');

if ~isprop(out, 'imuData')

    error('out.imuData does not exist. Run the Simulink model first.');

end

imuData = out.imuData;

[numSamples, numChannels] = size(imuData);

fprintf('Total IMU samples: %d\n', numSamples);
fprintf('IMU channels: %d\n', numChannels);


if numChannels ~= 6

    error('Expected 6 IMU channels, but found %d.', numChannels);

end

if numSamples < 4

    error('At least 4 IMU samples are required.');

end


%% Select one representative 4-sample window

% 301 = approximately 30 seconds
% with a fixed step size of 0.1 seconds.

imuStartSample = 301;

if imuStartSample + 3 > numSamples

    error('Selected IMU window exceeds available samples.');

end


imuWindow = imuData( ...
    imuStartSample:imuStartSample+3, :);


fprintf('\nUsing IMU samples %d-%d\n', ...
    imuStartSample, imuStartSample+3);


%% Display selected IMU window

fprintf('\n');
fprintf('--------------------------------------------\n');
fprintf('       SELECTED IMU WINDOW\n');
fprintf('--------------------------------------------\n');

fprintf('Sample\tAccX\t\tAccY\t\tAccZ\t\tGyroX\t\tGyroY\t\tGyroZ\n');

for k = 1:4

    fprintf( ...
        '%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        k, ...
        imuWindow(k,1), ...
        imuWindow(k,2), ...
        imuWindow(k,3), ...
        imuWindow(k,4), ...
        imuWindow(k,5), ...
        imuWindow(k,6));

end

fprintf('--------------------------------------------\n');


%% Create API samples

samples = [ ...

    struct( ...
        'AccX',imuWindow(1,1), ...
        'AccY',imuWindow(1,2), ...
        'AccZ',imuWindow(1,3), ...
        'GyroX',imuWindow(1,4), ...
        'GyroY',imuWindow(1,5), ...
        'GyroZ',imuWindow(1,6)), ...

    struct( ...
        'AccX',imuWindow(2,1), ...
        'AccY',imuWindow(2,2), ...
        'AccZ',imuWindow(2,3), ...
        'GyroX',imuWindow(2,4), ...
        'GyroY',imuWindow(2,5), ...
        'GyroZ',imuWindow(2,6)), ...

    struct( ...
        'AccX',imuWindow(3,1), ...
        'AccY',imuWindow(3,2), ...
        'AccZ',imuWindow(3,3), ...
        'GyroX',imuWindow(3,4), ...
        'GyroY',imuWindow(3,5), ...
        'GyroZ',imuWindow(3,6)), ...

    struct( ...
        'AccX',imuWindow(4,1), ...
        'AccY',imuWindow(4,2), ...
        'AccZ',imuWindow(4,3), ...
        'GyroX',imuWindow(4,4), ...
        'GyroY',imuWindow(4,5), ...
        'GyroZ',imuWindow(4,6)) ...

];


%% Create payload

imu_payload.samples = samples;


%% Send IMU data to FastAPI

fprintf('\nSending IMU window to FastAPI...\n');

imu_url = ...
    'http://127.0.0.1:8000/api/v1/imu/predict';


imu_response = webwrite( ...
    imu_url, ...
    imu_payload, ...
    options);


%% Convert class number to behaviour

switch imu_response.predicted_class

    case 1
        behaviour = 'Normal Driving';

    case 2
        behaviour = 'Harsh Acceleration';

    case 3
        behaviour = 'Harsh Braking';

    case 4
        behaviour = 'Sharp Cornering';

    otherwise
        behaviour = 'Unknown';

end


%% Display IMU result

fprintf('\n');
fprintf('--------------------------------------------\n');
fprintf('          IMU AI RESULT\n');
fprintf('--------------------------------------------\n');

fprintf('Model: %s\n', ...
    imu_response.model);

fprintf('Predicted class: %d\n', ...
    imu_response.predicted_class);

fprintf('Features: %d\n', ...
    imu_response.feature_count);

fprintf('Window size: %d\n', ...
    imu_response.window_size);

fprintf('Driving Behaviour: %s\n', ...
    behaviour);

fprintf('--------------------------------------------\n');


%% ============================================================
% 4. FINAL DRIVEGUARD-X RESULT
%% ============================================================

fprintf('\n\n');
fprintf('============================================================\n');
fprintf('                 DRIVEGUARD-X FINAL RESULT                  \n');
fprintf('============================================================\n');

fprintf('\n');

fprintf('BATTERY AI\n');
fprintf('  Model           : %s\n', ...
    battery_response.model);

fprintf('  Battery SOH     : %.2f %%\n', ...
    battery_response.soh_percent);


fprintf('\n');

fprintf('IMU AI\n');
fprintf('  Model           : %s\n', ...
    imu_response.model);

fprintf('  Window Size     : %d samples\n', ...
    imu_response.window_size);

fprintf('  Features        : %d\n', ...
    imu_response.feature_count);

fprintf('  Predicted Class : %d\n', ...
    imu_response.predicted_class);

fprintf('  Behaviour       : %s\n', ...
    behaviour);


fprintf('\n');
fprintf('============================================================\n');
fprintf('              AI INFERENCE COMPLETE                         \n');
fprintf('============================================================\n');

fprintf('\nDriveGuard-X simulation and AI inference completed successfully.\n');