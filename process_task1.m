%% IEEE Signal Processing Cup 2026 - Task 1: Anechoic Chamber Simulation
% Optimized for Edge Devices - MATLAB Coder Compatible
%
% Author: Signal Processing Cup 2026 Participant
% Date: January 2026
%
% This script implements audio zooming in an ideal anechoic environment
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

%% Room Configuration (Anechoic Chamber)
room.dimensions = single([4.9, 4.9, 4.9]);  % Room size [width, length, height] (m)
room.rt60 = single(0.0);                     % Reverberation time (s) - anechoic

%% Microphone Array Configuration
% 2-element Uniform Linear Array
mic.positions = single([2.41, 2.45, 1.5;    % Mic 1 position
                        2.49, 2.45, 1.5]);  % Mic 2 position
mic.num_elements = int32(size(mic.positions, 1));
mic.center = mean(mic.positions, 1);
mic.spacing = norm(mic.positions(1,:) - mic.positions(2,:));

fprintf('Microphone Array Configuration:\n');
fprintf('  Number of microphones: %d\n', mic.num_elements);
fprintf('  Spacing: %.3f m (%.1f cm)\n', mic.spacing, mic.spacing*100);
fprintf('  Center position: [%.2f, %.2f, %.2f] m\n', mic.center);

%% Source Configuration
% Target Source (Male Speech at 90 degrees)
source1.position = single([2.45, 3.45, 1.5]);
source1.azimuth = single(90);
source1.elevation = single(0);
source1.type = 'target';

% Interference Source (Female Speech at 40 degrees)
source2.position = single([3.22, 3.06, 1.5]);
source2.azimuth = single(40);
source2.elevation = single(0);
source2.type = 'interference';

% Verify calculated angles
[source1_calc_az, source1_calc_el] = calculateAngles(source1.position, mic.center);
[source2_calc_az, source2_calc_el] = calculateAngles(source2.position, mic.center);

fprintf('\nSource Configuration:\n');
fprintf('  Target (Source 1):\n');
fprintf('    Position: [%.2f, %.2f, %.2f] m\n', source1.position);
fprintf('    Calculated Azimuth: %.1f° (Expected: %.1f°)\n', source1_calc_az, source1.azimuth);
fprintf('  Interference (Source 2):\n');
fprintf('    Position: [%.2f, %.2f, %.2f] m\n', source2.position);
fprintf('    Calculated Azimuth: %.1f° (Expected: %.1f°)\n', source2_calc_az, source2.azimuth);

%% Load Audio Signals
fprintf('\nLoading audio signals...\n');

% Load target signal (male speech)
target_file = fullfile('musan', 'speech', 'librivox', 'speech-librivox-0000.wav');
if ~exist(target_file, 'file')
    error('Target audio file not found: %s', target_file);
end
[target_signal, fs_target] = audioread(target_file);
target_signal = single(target_signal);
target_signal = target_signal / max(abs(target_signal));  % Normalize

% Load interference signal (female speech)
interf_file = fullfile('musan', 'speech', 'librivox', 'speech-librivox-0100.wav');
if ~exist(interf_file, 'file')
    error('Interference audio file not found: %s', interf_file);
end
[interf_signal, fs_interf] = audioread(interf_file);
interf_signal = single(interf_signal);
interf_signal = interf_signal / max(abs(interf_signal));  % Normalize

% Verify sampling rates
if fs_target ~= config.fs || fs_interf ~= config.fs
    error('Audio file sampling rate mismatch. Expected %d Hz', config.fs);
end

% Align signal lengths (truncate to minimum length)
signal_length = min(length(target_signal), length(interf_signal));
target_signal = target_signal(1:signal_length);
interf_signal = interf_signal(1:signal_length);

fprintf('  Signal length: %.2f seconds (%d samples)\n', ...
    signal_length/config.fs, signal_length);

%% Simulate Room Acoustics (Anechoic - Direct Path Only)
fprintf('\nSimulating anechoic room acoustics...\n');

% For anechoic chamber, we only model time delays (no reflections)
% Calculate time delays for each microphone
delays1 = calculateTimeDelays(source1.position, mic.positions, config.c);
delays2 = calculateTimeDelays(source2.position, mic.positions, config.c);

% Apply delays to create microphone signals
target_mic_signals = applyDelays(target_signal, delays1, config.fs);
interf_mic_signals = applyDelays(interf_signal, delays2, config.fs);

%% Create Mixture with Specified SIR and SNR
fprintf('Creating mixture (SIR = %.1f dB, SNR = %.1f dB)...\n', ...
    config.desired_sir, config.desired_snr);

% Adjust interference level for desired SIR
sir_linear = 10^(config.desired_sir / 20);
interf_scaled = interf_mic_signals / sir_linear;

% Add interference to target
mixture_signal = target_mic_signals + interf_scaled;

% Add white Gaussian noise for desired SNR
signal_power = mean(mixture_signal.^2, 1);
noise_power = signal_power / (10^(config.desired_snr / 10));
noise = sqrt(noise_power) .* randn(size(mixture_signal), 'single');
mixture_signal = mixture_signal + noise;

fprintf('  Mixture signal power: %.6f\n', mean(mixture_signal(:).^2));

%% Audio Zooming Processing
fprintf('\nApplying audio zooming beamforming...\n');

% Apply delay-and-sum beamforming (optimized for edge devices)
processed_signal_ds = delayAndSumBeamformer(mixture_signal, ...
    source1.azimuth, mic.spacing, config.c, config.fs);

% Apply MVDR beamforming for comparison
processed_signal_mvdr = mvdrBeamformer(mixture_signal, ...
    source1.azimuth, mic.spacing, config.c, config.fs, config.fft_size);

fprintf('  Processing complete.\n');

%% Evaluation Metrics
fprintf('\nCalculating evaluation metrics...\n');

% Metrics for Delay-and-Sum Beamformer
metrics_ds = calculateMetrics(target_signal, processed_signal_ds, config.fs);

% Metrics for MVDR Beamformer
metrics_mvdr = calculateMetrics(target_signal, processed_signal_mvdr, config.fs);

% Display results
fprintf('\n=== TASK 1 RESULTS (Anechoic Chamber) ===\n');
fprintf('\nDelay-and-Sum Beamformer:\n');
fprintf('  OSINR: %.2f dB\n', metrics_ds.osinr);
fprintf('  ViSQOL: %.3f\n', metrics_ds.visqol);
fprintf('  STOI: %.3f\n', metrics_ds.stoi);

fprintf('\nMVDR Beamformer:\n');
fprintf('  OSINR: %.2f dB\n', metrics_mvdr.osinr);
fprintf('  ViSQOL: %.3f\n', metrics_mvdr.visqol);
fprintf('  STOI: %.3f\n', metrics_mvdr.stoi);

%% Save Results
fprintf('\nSaving results...\n');

% Save data for submission
save('Task1_Anechoic_5dB.mat', ...
    'target_signal', 'interf_signal', 'mixture_signal', ...
    'processed_signal_ds', 'processed_signal_mvdr', ...
    'metrics_ds', 'metrics_mvdr', ...
    'config', 'mic', 'source1', 'source2', '-v7.3');

% Save audio files
audiowrite('target_signal.wav', double(target_signal), config.fs);
audiowrite('interference_signal1.wav', double(interf_signal), config.fs);
audiowrite('mixture_signal.wav', double(mean(mixture_signal, 2)), config.fs);
audiowrite('processed_signal_ds.wav', double(processed_signal_ds), config.fs);
audiowrite('processed_signal_mvdr.wav', double(processed_signal_mvdr), config.fs);

fprintf('Results saved successfully.\n');
fprintf('\n=== Task 1 Complete ===\n');

%% Helper Functions

function [azimuth, elevation] = calculateAngles(source_pos, mic_center)
    % Calculate azimuth and elevation angles from positions
    v = source_pos - mic_center;
    [az_rad, el_rad, ~] = cart2sph(v(1), v(2), v(3));
    azimuth = single(rad2deg(az_rad));
    elevation = single(rad2deg(el_rad));
end

function delays = calculateTimeDelays(source_pos, mic_positions, c)
    % Calculate time delays for each microphone relative to the first mic
    num_mics = size(mic_positions, 1);
    distances = zeros(num_mics, 1, 'single');

    for i = 1:num_mics
        distances(i) = norm(source_pos - mic_positions(i, :));
    end

    % Delays relative to first microphone
    delays = single((distances - distances(1)) / c);
end

function mic_signals = applyDelays(signal, delays, fs)
    % Apply time delays to signal for each microphone
    num_mics = length(delays);
    signal_length = length(signal);
    mic_signals = zeros(signal_length, num_mics, 'single');

    for i = 1:num_mics
        delay_samples = round(delays(i) * fs);
        if delay_samples >= 0
            % Positive delay - shift signal forward
            if delay_samples < signal_length
                mic_signals(delay_samples+1:end, i) = signal(1:end-delay_samples);
            end
        else
            % Negative delay - shift signal backward
            delay_samples = abs(delay_samples);
            if delay_samples < signal_length
                mic_signals(1:end-delay_samples, i) = signal(delay_samples+1:end);
            end
        end
    end
end
