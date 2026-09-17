%% ============================================================
% DRIVEGUARD-X
% REPRESENTATIVE IMU WINDOW TEST
%
% Simulink -> IMU -> 4-sample windows -> FastAPI -> XGBoost
%
% Sampling time = 0.1 s
% Simulation time = 60 s
%
% Test windows:
% 0.0 s   -> samples 1-4
% 10.0 s  -> samples 101-104
% 20.0 s  -> samples 201-204
% 30.0 s  -> samples 301-304
% 40.0 s  -> samples 401-404
% 50.0 s  -> samples 501-504
%% ============================================================

clc;

fprintf('\n');
fprintf('====================================================\n');
fprintf('       DRIVEGUARD-X REPRESENTATIVE IMU TEST        \n');
fprintf('====================================================\n');


%% 1. Check simulation output

if ~exist('out','var')
    error('Variable "out" does not exist. Run the Simulink model first.');
end

if ~isprop(out,'imuData')
    error('out.imuData does not exist. Run the Simulink simulation first.');
end

imuData = out.imuData;


%% 2. Check IMU dimensions

[numSamples, numChannels] = size(imuData);

fprintf('\nTotal IMU samples : %d\n', numSamples);
fprintf('IMU channels      : %d\n', numChannels);

if numChannels ~= 6
    error('Expected 6 IMU channels, but found %d.', numChannels);
end

if numSamples < 504
    error(['At least 504 IMU samples are required for the ', ...
           '0, 10, 20, 30, 40 and 50 second windows.']);
end


%% 3. Define representative windows

% Starting sample of each 4-sample window

testStarts = [1 101 201 301 401 501];

% Corresponding simulation times

testTimes = [0 10 20 30 40 50];


%% 4. FastAPI configuration

url = 'http://127.0.0.1:8000/api/v1/imu/predict';

options = weboptions( ...
    'MediaType','application/json', ...
    'Timeout',30);


%% 5. Storage for results

numTests = length(testStarts);

predictedClasses = zeros(numTests,1);


%% 6. Test each representative window

fprintf('\n');
fprintf('Testing representative windows...\n');
fprintf('----------------------------------------------------\n');

for k = 1:numTests

    startIndex = testStarts(k);
    endIndex = startIndex + 3;

    window = imuData(startIndex:endIndex,:);


    %% Create API samples

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


    %% Store prediction

    predictedClasses(k) = response.predicted_class;


    %% Convert class to readable behaviour

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
        'Time %2d s | Samples %3d-%3d | Class %d | %s\n', ...
        testTimes(k), ...
        startIndex, ...
        endIndex, ...
        response.predicted_class, ...
        behaviour);

end


%% 7. Create results table

results = table( ...
    testTimes(:), ...
    testStarts(:), ...
    (testStarts(:) + 3), ...
    predictedClasses, ...
    'VariableNames', ...
    {'TimeSeconds','StartSample','EndSample','PredictedClass'});


%% 8. Display final table

fprintf('\n');
fprintf('====================================================\n');
fprintf('                  TEST RESULTS                     \n');
fprintf('====================================================\n');

disp(results);


%% 9. Count classes

class1Count = sum(predictedClasses == 1);
class2Count = sum(predictedClasses == 2);
class3Count = sum(predictedClasses == 3);
class4Count = sum(predictedClasses == 4);


%% 10. Display summary

fprintf('====================================================\n');
fprintf('                    SUMMARY                         \n');
fprintf('====================================================\n');

fprintf('Total windows tested : %d\n\n', numTests);

fprintf('Class 1 - Normal Driving     : %d\n', class1Count);
fprintf('Class 2 - Harsh Acceleration : %d\n', class2Count);
fprintf('Class 3 - Harsh Braking      : %d\n', class3Count);
fprintf('Class 4 - Sharp Cornering    : %d\n', class4Count);

fprintf('====================================================\n');
fprintf('             IMU TEST COMPLETE                     \n');
fprintf('====================================================\n');