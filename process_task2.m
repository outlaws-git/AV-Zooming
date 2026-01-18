%% IEEE Signal Processing Cup 2026 - Task 2: Reverberant Room Simulation
% Optimized for Edge Devices - MATLAB Coder Compatible
%
% Author: Signal Processing Cup 2026 Participant
% Date: January 2026
%
% This script implements audio zooming in a realistic reverberant environment
% with optimization for resource-constrained edge devices.

%% Cleanup and initialization
clc; clear; close all;

%% Configuration Parameters
% Use single precision for memory efficiency on edge devices
config.fs = single(16000);              % Sampling rate (Hz)
config.c = single(340);                 % Speed of sound (m/s)
config.desired_sir = single(0);         % Signal-to-Interference Ratio (dB)
config.desired_snr = single(5);         % Signal-to-Noise Ratio (dB)
config.fft_size = int32(512);           % FFT size for frequency domain processing
config.overlap = single(0.5);           % Frame overlap ratio

%% Room Configuration (Reverberant Room)
room.dimensions = single([4.9, 4.9, 4.9]);  % Room size [width, length, height] (m)
room.rt60 = single(0.5);                     % Reverberation time (s) - moderate
room.reflection_order = int32(10);           % Number of reflections for ISM

% Calculate absorption coefficients for desired RT60
% Using Sabine's equation: RT60 = 0.161 * V / (S * α)
room.volume = prod(room.dimensions);
room.surface_area = 2 * (room.dimensions(1)*room.dimensions(2) + ...
                         room.dimensions(1)*room.dimensions(3) + ...
                         room.dimensions(2)*room.dimensions(3));

% Calculate average absorption coefficient
room.alpha = single(0.161 * room.volume / (room.surface_area * room.rt60));
room.alpha = min(room.alpha, 0.95);  % Cap at 0.95 for stability

fprintf('Room Configuration (Reverberant):\n');
fprintf('  Dimensions: %.1f x %.1f x %.1f m\n', room.dimensions);
fprintf('  RT60: %.2f s\n', room.rt60);
fprintf('  Absorption coefficient: %.3f\n', room.alpha);

%% Microphone Array Configuration
% 2-element Uniform Linear Array (same as Task 1)
mic.positions = single([2.41, 2.45, 1.5;    % Mic 1 position
                        2.49, 2.45, 1.5]);  % Mic 2 position
mic.num_elements = int32(size(mic.positions, 1));
mic.center = mean(mic.positions, 1);
mic.spacing = norm(mic.positions(1,:) - mic.positions(2,:));

fprintf('\nMicrophone Array Configuration:\n');
fprintf('  Number of microphones: %d\n', mic.num_elements);
fprintf('  Spacing: %.3f m (%.1f cm)\n', mic.spacing, mic.spacing*100);

%% Source Configuration (same as Task 1)
source1.position = single([2.45, 3.45, 1.5]);
source1.azimuth = single(90);
source2.position = single([3.22, 3.06, 1.5]);
source2.azimuth = single(40);

fprintf('\nSource Configuration:\n');
fprintf('  Target: %.1f° azimuth\n', source1.azimuth);
fprintf('  Interference: %.1f° azimuth\n', source2.azimuth);

%% Load Audio Signals
fprintf('\nLoading audio signals...\n');

% Load target signal (male speech)
target_file = fullfile('musan', 'speech', 'librivox', 'speech-librivox-0000.wav');
[target_signal, fs_target] = audioread(target_file);
target_signal = single(target_signal);
target_signal = target_signal / max(abs(target_signal));

% Load interference signal (female speech)
interf_file = fullfile('musan', 'speech', 'librivox', 'speech-librivox-0100.wav');
[interf_signal, fs_interf] = audioread(interf_file);
interf_signal = single(interf_signal);
interf_signal = interf_signal / max(abs(interf_signal));

% Align signal lengths
signal_length = min(length(target_signal), length(interf_signal));
target_signal = target_signal(1:signal_length);
interf_signal = interf_signal(1:signal_length);

fprintf('  Signal length: %.2f seconds\n', signal_length/config.fs);

%% Simulate Room Acoustics with Reverberation
fprintf('\nSimulating reverberant room acoustics using Image Source Method...\n');

% Generate Room Impulse Responses (RIRs) for each source-microphone pair
fprintf('  Generating RIRs for target source...\n');
target_rirs = generateRIR(source1.position, mic.positions, room, config.fs);

fprintf('  Generating RIRs for interference source...\n');
interf_rirs = generateRIR(source2.position, mic.positions, room, config.fs);

% Convolve source signals with RIRs
fprintf('  Applying room acoustics...\n');
target_mic_signals = applyRIR(target_signal, target_rirs);
interf_mic_signals = applyRIR(interf_signal, interf_rirs);

%% Create Mixture with Specified SIR and SNR
fprintf('Creating mixture (SIR = %.1f dB, SNR = %.1f dB)...\n', ...
    config.desired_sir, config.desired_snr);

% Adjust interference level for desired SIR
sir_linear = 10^(config.desired_sir / 20);
interf_scaled = interf_mic_signals / sir_linear;

% Add interference to target
mixture_signal = target_mic_signals + interf_scaled;

% Add white Gaussian noise for desired SNR (including reverberation)
signal_power = mean(mixture_signal.^2, 1);
noise_power = signal_power / (10^(config.desired_snr / 10));
noise = sqrt(noise_power) .* randn(size(mixture_signal), 'single');
mixture_signal = mixture_signal + noise;

%% Audio Zooming Processing with Dereverberation
fprintf('\nApplying audio zooming with dereverberation...\n');

% Apply delay-and-sum beamforming
processed_signal_ds = delayAndSumBeamformer(mixture_signal, ...
    source1.azimuth, mic.spacing, config.c, config.fs);

% Apply MVDR beamforming
processed_signal_mvdr = mvdrBeamformer(mixture_signal, ...
    source1.azimuth, mic.spacing, config.c, config.fs, config.fft_size);

% Optional: Apply simple spectral subtraction for dereverberation
processed_signal_ds_dereverb = spectralDereverberation(processed_signal_ds, config.fs);
processed_signal_mvdr_dereverb = spectralDereverberation(processed_signal_mvdr, config.fs);

fprintf('  Processing complete.\n');

%% Evaluation Metrics
fprintf('\nCalculating evaluation metrics...\n');

% Metrics for Delay-and-Sum (with and without dereverberation)
metrics_ds = calculateMetrics(target_signal, processed_signal_ds, config.fs);
metrics_ds_dereverb = calculateMetrics(target_signal, processed_signal_ds_dereverb, config.fs);

% Metrics for MVDR (with and without dereverberation)
metrics_mvdr = calculateMetrics(target_signal, processed_signal_mvdr, config.fs);
metrics_mvdr_dereverb = calculateMetrics(target_signal, processed_signal_mvdr_dereverb, config.fs);

%% Display Results
fprintf('\n=== TASK 2 RESULTS (Reverberant Room, RT60 = %.2f s) ===\n', room.rt60);

fprintf('\nDelay-and-Sum Beamformer:\n');
fprintf('  OSINR: %.2f dB\n', metrics_ds.osinr);
fprintf('  ViSQOL: %.3f\n', metrics_ds.visqol);
fprintf('  STOI: %.3f\n', metrics_ds.stoi);

fprintf('\nDelay-and-Sum + Dereverberation:\n');
fprintf('  OSINR: %.2f dB\n', metrics_ds_dereverb.osinr);
fprintf('  ViSQOL: %.3f\n', metrics_ds_dereverb.visqol);
fprintf('  STOI: %.3f\n', metrics_ds_dereverb.stoi);

fprintf('\nMVDR Beamformer:\n');
fprintf('  OSINR: %.2f dB\n', metrics_mvdr.osinr);
fprintf('  ViSQOL: %.3f\n', metrics_mvdr.visqol);
fprintf('  STOI: %.3f\n', metrics_mvdr.stoi);

fprintf('\nMVDR + Dereverberation:\n');
fprintf('  OSINR: %.2f dB\n', metrics_mvdr_dereverb.osinr);
fprintf('  ViSQOL: %.3f\n', metrics_mvdr_dereverb.visqol);
fprintf('  STOI: %.3f\n', metrics_mvdr_dereverb.stoi);

%% Save Results
fprintf('\nSaving results...\n');

% Save data for submission
save('Task2_Reverberant_5dB.mat', ...
    'target_signal', 'interf_signal', 'mixture_signal', ...
    'processed_signal_mvdr_dereverb', ...
    'target_rirs', 'interf_rirs', ...
    'metrics_ds', 'metrics_mvdr', 'metrics_mvdr_dereverb', ...
    'config', 'room', 'mic', 'source1', 'source2', '-v7.3');

% Save audio files
audiowrite('target_signal.wav', double(target_signal), config.fs);
audiowrite('interference_signal1.wav', double(interf_signal), config.fs);
audiowrite('mixture_signal.wav', double(mean(mixture_signal, 2)), config.fs);
audiowrite('processed_signal.wav', double(processed_signal_mvdr_dereverb), config.fs);

fprintf('Results saved successfully.\n');
fprintf('\n=== Task 2 Complete ===\n');

%% Helper Functions

function rirs = generateRIR(source_pos, mic_positions, room, fs)
    % Generate Room Impulse Responses using simplified Image Source Method
    % Optimized for edge devices with reduced computational complexity

    num_mics = size(mic_positions, 1);
    rir_length = round(1.5 * room.rt60 * fs);  % RIR length based on RT60
    rirs = zeros(rir_length, num_mics, 'single');

    % Reflection coefficients (uniform for simplicity)
    beta = sqrt(1 - room.alpha);

    % Generate images up to specified order
    for mic_idx = 1:num_mics
        mic_pos = mic_positions(mic_idx, :);

        % Direct path
        distance = norm(source_pos - mic_pos);
        delay_samples = round(distance / 340 * fs);
        amplitude = 1 / (4 * pi * distance);

        if delay_samples < rir_length
            rirs(delay_samples + 1, mic_idx) = amplitude;
        end

        % First-order reflections (6 walls)
        walls = {'x-', 'x+', 'y-', 'y+', 'z-', 'z+'};
        for w = 1:length(walls)
            image_pos = reflectPoint(source_pos, walls{w}, room.dimensions);
            distance = norm(image_pos - mic_pos);
            delay_samples = round(distance / 340 * fs);
            amplitude = beta / (4 * pi * distance);

            if delay_samples < rir_length && delay_samples > 0
                rirs(delay_samples + 1, mic_idx) = ...
                    rirs(delay_samples + 1, mic_idx) + amplitude;
            end
        end
    end

    % Normalize RIRs
    for mic_idx = 1:num_mics
        rirs(:, mic_idx) = rirs(:, mic_idx) / max(abs(rirs(:, mic_idx)));
    end
end

function image_pos = reflectPoint(point, wall, room_dim)
    % Reflect a point across a wall
    image_pos = point;
    switch wall
        case 'x-'
            image_pos(1) = -point(1);
        case 'x+'
            image_pos(1) = 2 * room_dim(1) - point(1);
        case 'y-'
            image_pos(2) = -point(2);
        case 'y+'
            image_pos(2) = 2 * room_dim(2) - point(2);
        case 'z-'
            image_pos(3) = -point(3);
        case 'z+'
            image_pos(3) = 2 * room_dim(3) - point(3);
    end
end

function mic_signals = applyRIR(source_signal, rirs)
    % Convolve source signal with room impulse responses
    num_mics = size(rirs, 2);
    signal_length = length(source_signal) + size(rirs, 1) - 1;
    mic_signals = zeros(signal_length, num_mics, 'single');

    for mic_idx = 1:num_mics
        mic_signals(:, mic_idx) = conv(source_signal, rirs(:, mic_idx));
    end
end
