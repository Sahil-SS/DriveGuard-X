%% DRIVEGUARD-X SIMULINK IMU -> FASTAPI

clc;

fprintf('\n');
fprintf('============================================\n');
fprintf('       DRIVEGUARD-X SIMULINK IMU AI        \n');
fprintf('============================================\n');

%% 1. Get IMU data from Simulink

if ~exist('out','var')
    error('Simulation output "out" does not exist. Run the Simulink model first.');
end

imuData = out.imuData;

[numSamples, numChannels] = size(imuData);

fprintf('\nTotal IMU samples : %d\n', numSamples);
fprintf('IMU channels      : %d\n', numChannels);

if numChannels ~= 6
    error('Expected 6 IMU channels, but found %d.', numChannels);
end

if numSamples < 4
    error('At least 4 IMU samples are required.');
end

%% 2. Take four consecutive samples

window = imuData(56:59,:);

fprintf('\n4-sample IMU window:\n');
fprintf('------------------------------------------------------------\n');
fprintf('Sample\tAccX\t\tAccY\t\tAccZ\t\tGyroX\t\tGyroY\t\tGyroZ\n');
fprintf('------------------------------------------------------------\n');

for i = 1:4
    fprintf('%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        i, ...
        window(i,1), ...
        window(i,2), ...
        window(i,3), ...
        window(i,4), ...
        window(i,5), ...
        window(i,6));
end

%% 3. Build API payload

samples = [ ...
    struct('AccX',window(1,1),'AccY',window(1,2),'AccZ',window(1,3), ...
           'GyroX',window(1,4),'GyroY',window(1,5),'GyroZ',window(1,6)), ...

    struct('AccX',window(2,1),'AccY',window(2,2),'AccZ',window(2,3), ...
           'GyroX',window(2,4),'GyroY',window(2,5),'GyroZ',window(2,6)), ...

    struct('AccX',window(3,1),'AccY',window(3,2),'AccZ',window(3,3), ...
           'GyroX',window(3,4),'GyroY',window(3,5),'GyroZ',window(3,6)), ...

    struct('AccX',window(4,1),'AccY',window(4,2),'AccZ',window(4,3), ...
           'GyroX',window(4,4),'GyroY',window(4,5),'GyroZ',window(4,6)) ...
];

payload.samples = samples;

%% 4. Send to FastAPI

fprintf('\nSending Simulink IMU window to FastAPI...\n');

url = 'http://127.0.0.1:8000/api/v1/imu/predict';

options = weboptions( ...
    'MediaType','application/json', ...
    'Timeout',30);

response = webwrite(url,payload,options);

%% 5. Display result

fprintf('\n');
fprintf('============================================\n');
fprintf('       DRIVEGUARD-X IMU AI RESULT          \n');
fprintf('============================================\n');

fprintf('Model: %s\n',response.model);
fprintf('Predicted class: %d\n',response.predicted_class);
fprintf('Features: %d\n',response.feature_count);
fprintf('Window size: %d\n',response.window_size);

switch response.predicted_class

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

fprintf('Driving Behaviour: %s\n',behaviour);

fprintf('============================================\n');
fprintf('\nIMU AI prediction completed successfully.\n');