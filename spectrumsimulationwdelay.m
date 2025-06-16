% Complete code for generating spectrum occupancy dataset with transition delays

%% 1. Generate Primary User Activity with Transition Delays
% Number of time slots
N = 1000;

% States: 0 = Idle, 1 = Busy, 2 = Hybrid
states = [0, 1, 2];

% Transition Probability Matrix
P = [0.7, 0.1, 0.2;   % From Idle
     0.2, 0.6, 0.2;   % From Busy
     0.3, 0.3, 0.4];  % From Hybrid

% Transition delay (in time slots)
transition_delay = 3;  % Adjust based on your requirements

% Initialize variables
PU_state = zeros(1, N);
current_state = randi([0, 2]);  % Initial state
PU_state(1) = current_state;
transition_counter = 0;
target_state = current_state;

% Simulate with transition delays
for t = 2:N
    if transition_counter > 0
        % In transition: maintain current state
        PU_state(t) = PU_state(t-1);
        transition_counter = transition_counter - 1;
        
        % Apply new state when delay completes
        if transition_counter == 0
            PU_state(t) = target_state;
            current_state = target_state;
        end
    else
        % Determine next state using transition matrix
        next_state = randsample(states, 1, true, P(current_state + 1, :));
        
        if next_state ~= current_state
            % Initiate transition
            transition_counter = transition_delay;
            target_state = next_state;
            PU_state(t) = current_state;  % Stay in current state during transition
        else
            % No state change needed
            PU_state(t) = current_state;
        end
    end
end

% Plot the result with transition delays
figure;
stairs(PU_state, 'LineWidth', 1.5);
ylim([-0.5, 2.5]);
yticks([0 1 2]);
yticklabels({'Idle', 'Busy', 'Hybrid'});
xlabel('Time Slot');
ylabel('Primary User State');
title('Primary User Activity with Transition Delays');
grid on;

%% 2. Extract Features and Create Dataset

% Create timestamped dataset
timesteps = 1:N;
PU_state_data = PU_state';

% Feature 1: Time since last state change
time_since_change = zeros(N, 1);
count = 0;
prev_state = PU_state(1);
for i = 1:N
    if PU_state(i) == prev_state
        count = count + 1;
    else
        count = 0;
        prev_state = PU_state(i);
    end
    time_since_change(i) = count;
end

% Feature 2: Create sliding window features (last 5 states)
window_size = 5;
window_features = zeros(N, window_size);
for i = 1:N
    for j = 1:window_size
        if i-j+1 > 0
            window_features(i,j) = PU_state(i-j+1);
        end
    end
end

% Feature 3: Calculate transition probabilities in local windows
window_length = 50;
local_transition_probs = zeros(N, 9); % 3x3 transition matrix flattened
for i = window_length:N
    window_states = PU_state(i-window_length+1:i);
    
    % Calculate local transition probabilities
    local_P = zeros(3,3);
    for j = 2:length(window_states)
        from_state = window_states(j-1) + 1; % +1 for MATLAB indexing
        to_state = window_states(j) + 1;     % +1 for MATLAB indexing
        local_P(from_state, to_state) = local_P(from_state, to_state) + 1;
    end
    
    % Normalize
    for j = 1:3
        if sum(local_P(j,:)) > 0
            local_P(j,:) = local_P(j,:) / sum(local_P(j,:));
        end
    end
    
    % Flatten to feature vector
    local_transition_probs(i,:) = reshape(local_P, 1, 9);
end

% Feature 4: Channel occupancy rate in last window
occupancy_rate = zeros(N, 3);
for i = window_length:N
    window_states = PU_state(i-window_length+1:i);
    for j = 0:2
        occupancy_rate(i,j+1) = sum(window_states == j) / window_length;
    end
end

% Feature 5: Is the system currently in transition? (binary feature)
in_transition = zeros(N, 1);
for i = 2:N
    if PU_state(i) == PU_state(i-1) && time_since_change(i) < transition_delay
        in_transition(i) = 1;
    end
end

% Feature 6: Next state prediction target
next_state = [PU_state(2:end), PU_state(end)];

% Combine all features
base_features = [timesteps', PU_state_data, time_since_change, in_transition, window_features];
advanced_features = [local_transition_probs, occupancy_rate];
targets = next_state';

% Complete dataset with all features
full_dataset = [base_features, advanced_features, targets];

%% 3. Save Dataset to CSV

% Define column headers
base_headers = {'TimeSlot', 'PU_State', 'TimeSinceChange', 'InTransition'};
window_headers = cell(1, window_size);
for i = 1:window_size
    window_headers{i} = ['PreviousState_', num2str(i)];
end

transition_headers = cell(1, 9);
for i = 1:3
    for j = 1:3
        transition_headers{(i-1)*3+j} = sprintf('Trans_From%d_To%d', i-1, j-1);
    end
end

occupancy_headers = cell(1, 3);
for i = 1:3
    occupancy_headers{i} = sprintf('OccupancyRate_State%d', i-1);
end

target_header = {'NextState'};

% Combine all headers
all_headers = [base_headers, window_headers, transition_headers, occupancy_headers, target_header];

% Create table and save to CSV
T = array2table(full_dataset, 'VariableNames', all_headers);
writetable(T, 'spectrum_occupancy_dataset.csv');

% Split into training and testing sets (80/20 split)
split_point = floor(0.8 * N);
train_data = full_dataset(1:split_point, :);
test_data = full_dataset(split_point+1:end, :);

% Save training and testing datasets
train_table = array2table(train_data, 'VariableNames', all_headers);
test_table = array2table(test_data, 'VariableNames', all_headers);
writetable(train_table, 'spectrum_train_data.csv');
writetable(test_table, 'spectrum_test_data.csv');

fprintf('Dataset generation complete!\n');
fprintf('Full dataset: %d samples with %d features\n', size(full_dataset, 1), size(full_dataset, 2)-1);
fprintf('Training data: %d samples\n', size(train_data, 1));
fprintf('Testing data: %d samples\n', size(test_data, 1));
fprintf('Files saved: spectrum_occupancy_dataset.csv, spectrum_train_data.csv, spectrum_test_data.csv\n');
