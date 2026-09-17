%% ============================================================
% DRIVEGUARD-X
% MODE 1 - TEST ALL IMU ROLLING WINDOWS
%
% 68 samples -> 65 possible 4-sample windows
%
% Window:
% 1-4
% 2-5
% 3-6
% ...
% 65-68
%% ============================================================

clc;

fprintf('\n');
fprintf('====================================================\n');
fprintf('     DRIVEGUARD-X - IMU WINDOW TEST         \n');
fprintf('====================================================\n');


%% 1. Get Simulink IMU data

if ~exist('out','var')
    error('Simulation output "out" does not exist. Run Simulink first.');
end

imuData = out.imuData;


%% 2. Check dimensions

[numSamples, numChannels] = size(imuData);

fprintf('\nTotal IMU samples : %d\n', numSamples);
fprintf('IMU channels      : %d\n', numChannels);

if numChannels ~= 6
    error('Expected 6 IMU channels.');
end

if numSamples < 4
    error('At least 4 samples are required.');
end


%% 3. Calculate number of rolling windows

numWindows = numSamples - 4 + 1;

fprintf('Total 4-sample windows: %d\n', numWindows);


%% 4. FastAPI settings

url = 'http://127.0.0.1:8000/api/v1/imu/predict';

options = weboptions( ...
    'MediaType','application/json', ...
    'Timeout',30);


%% 5. Storage for results

predictedClasses = zeros(numWindows,1);

windowStarts = zeros(numWindows,1);

windowEnds = zeros(numWindows,1);


%% 6. Test every rolling window

fprintf('\n');
fprintf('Testing all rolling windows...\n');
fprintf('----------------------------------------------------\n');

for w = 1:numWindows

    %% Current window

    startIndex = w;

    endIndex = w + 3;

    window = imuData(startIndex:endIndex,:);


    %% Create four API samples

    samples = [ ...

        struct( ...
            'AccX',window(1,1), ...
            'AccY',window(1,2), ...
            'AccZ',window(1,3), ...
            'GyroX',window(1,4), ...
            'GyroY',window(1,5), ...
            'GyroZ',window(1,6)), ...

        struct( ...
            'AccX',window(2,1), ...
            'AccY',window(2,2), ...
            'AccZ',window(2,3), ...
            'GyroX',window(2,4), ...
            'GyroY',window(2,5), ...
            'GyroZ',window(2,6)), ...

        struct( ...
            'AccX',window(3,1), ...
            'AccY',window(3,2), ...
            'AccZ',window(3,3), ...
            'GyroX',window(3,4), ...
            'GyroY',window(3,5), ...
            'GyroZ',window(3,6)), ...

        struct( ...
            'AccX',window(4,1), ...
            'AccY',window(4,2), ...
            'AccZ',window(4,3), ...
            'GyroX',window(4,4), ...
            'GyroY',window(4,5), ...
            'GyroZ',window(4,6)) ...

    ];


    %% Create payload

    payload.samples = samples;


    %% Send to FastAPI

    response = webwrite( ...
        url, ...
        payload, ...
        options);


    %% Store result

    predictedClasses(w) = response.predicted_class;

    windowStarts(w) = startIndex;

    windowEnds(w) = endIndex;


    %% Convert class to text

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


    %% Display result

    fprintf( ...
        'Window %2d: Samples %2d-%2d -> Class %d -> %s\n', ...
        w, ...
        startIndex, ...
        endIndex, ...
        response.predicted_class, ...
        behaviour);


end


%% 7. Count each class

class1Count = sum(predictedClasses == 1);

class2Count = sum(predictedClasses == 2);

class3Count = sum(predictedClasses == 3);

class4Count = sum(predictedClasses == 4);


%% 8. Final summary

fprintf('\n');
fprintf('====================================================\n');
fprintf('                      IMU SUMMARY                   \n');
fprintf('====================================================\n');

fprintf('Total windows tested : %d\n\n', numWindows);

fprintf('Class 1 - Normal Driving     : %d windows\n', class1Count);

fprintf('Class 2 - Harsh Acceleration : %d windows\n', class2Count);

fprintf('Class 3 - Harsh Braking      : %d windows\n', class3Count);

fprintf('Class 4 - Sharp Cornering    : %d windows\n', class4Count);


%% 9. Create results table

results = table( ...
    windowStarts, ...
    windowEnds, ...
    predictedClasses, ...
    'VariableNames', ...
    {'StartSample','EndSample','PredictedClass'});


%% 10. Display table

fprintf('\n');
fprintf('Complete results table:\n\n');

disp(results);


fprintf('====================================================\n');
fprintf('                     TEST COMPLETE                  \n');
fprintf('====================================================\n');