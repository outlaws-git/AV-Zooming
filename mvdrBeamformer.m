function output_signal = mvdrBeamformer(mic_signals, target_azimuth, mic_spacing, c, fs, fft_size)
%MVDRBEAMFORMER Minimum Variance Distortionless Response beamformer for edge devices
%
% Implements MVDR beamforming in the frequency domain with optimizations
% for resource-constrained edge devices.
%
% Inputs:
%   mic_signals     - Microphone signals [samples x num_mics]
%   target_azimuth  - Target direction in degrees
%   mic_spacing     - Distance between microphones (m)
%   c               - Speed of sound (m/s)
%   fs              - Sampling frequency (Hz)
%   fft_size        - FFT size for frequency domain processing
%
% Output:
%   output_signal   - Beamformed signal [samples x 1]
%
% Edge Optimizations:
%   - Single precision arithmetic
%   - Reduced FFT size for computational efficiency
%   - Diagonal loading for numerical stability
%   - Memory-efficient overlap-add processing

%% Type conversion
mic_signals = single(mic_signals);
target_azimuth = single(target_azimuth);
mic_spacing = single(mic_spacing);
c = single(c);
fs = single(fs);

%% Parameters
[num_samples, num_mics] = size(mic_signals);
hop_size = fft_size / 2;  % 50% overlap
num_freqs = fft_size / 2 + 1;

% Diagonal loading factor for regularization (important for edge stability)
diagonal_loading = single(1e-3);

%% Pre-compute steering vector for all frequencies
freq_bins = single(linspace(0, fs/2, num_freqs)');
steering_vectors = computeSteeringVectors(freq_bins, target_azimuth, ...
    mic_spacing, c, num_mics);

%% STFT Processing
num_frames = floor((num_samples - fft_size) / hop_size) + 1;
output_stft = zeros(num_freqs, num_frames, 'single');

% Hann window for smooth overlap-add
window = single(hann(fft_size, 'periodic'));

fprintf('  MVDR: Processing %d frames...', num_frames);

for frame_idx = 1:num_frames
    % Extract frame from all microphones
    start_idx = (frame_idx - 1) * hop_size + 1;
    end_idx = start_idx + fft_size - 1;

    if end_idx > num_samples
        break;
    end

    % Apply window and compute STFT for each microphone
    frame_stft = zeros(num_freqs, num_mics, 'single');
    for m = 1:num_mics
        frame = mic_signals(start_idx:end_idx, m) .* window;
        fft_result = fft(frame, fft_size);
        frame_stft(:, m) = fft_result(1:num_freqs);
    end

    % Process each frequency bin
    for freq_idx = 1:num_freqs
        % Extract frequency bin across all mics
        X = frame_stft(freq_idx, :).';  % [num_mics x 1]

        % Compute spatial covariance matrix with diagonal loading
        % R = E[X * X'] ≈ X * X' + δI
        R = (X * X') + diagonal_loading * eye(num_mics, 'single');

        % Get steering vector for this frequency
        a = steering_vectors(:, freq_idx);  % [num_mics x 1]

        % MVDR weight vector: w = (R^-1 * a) / (a' * R^-1 * a)
        % Use numerically stable computation
        try
            R_inv_a = R \ a;  % Solve R * x = a (more stable than inv(R) * a)
            denominator = a' * R_inv_a;

            if abs(denominator) > eps('single')
                w = R_inv_a / denominator;
            else
                % Fallback to delay-and-sum if ill-conditioned
                w = a / num_mics;
            end
        catch
            % If matrix inversion fails, use delay-and-sum weights
            w = a / num_mics;
        end

        % Apply beamforming weights
        output_stft(freq_idx, frame_idx) = w' * X;
    end

    % Progress indicator (every 100 frames)
    if mod(frame_idx, 100) == 0
        fprintf('.');
    end
end

fprintf(' done.\n');

%% Inverse STFT with overlap-add
output_signal = overlapAdd(output_stft, fft_size, hop_size);

% Trim to original length
output_signal = output_signal(1:num_samples);

% Normalize output
output_signal = output_signal / (max(abs(output_signal)) + eps('single'));

end

function steering_vectors = computeSteeringVectors(freq_bins, azimuth, d, c, num_mics)
%COMPUTESTEERINGVECTORS Compute steering vectors for all frequency bins
%
% For a uniform linear array, the steering vector at frequency f is:
%   a(f) = [1, exp(-j*2πf*τ), ..., exp(-j*2πf*(M-1)*τ)]'
% where τ = (d * sin(θ)) / c

    % Convert azimuth to angle from array axis
    theta = deg2rad(single(90 - azimuth));

    % Time delay between adjacent elements
    tau = single((d * sin(theta)) / c);

    % Initialize steering vectors [num_mics x num_freqs]
    num_freqs = length(freq_bins);
    steering_vectors = zeros(num_mics, num_freqs, 'single');

    % Compute for each frequency
    for f_idx = 1:num_freqs
        f = freq_bins(f_idx);
        phase_shift = -2 * pi * f * tau;

        for m = 1:num_mics
            steering_vectors(m, f_idx) = exp(1j * phase_shift * (m - 1));
        end
    end
end

function output_signal = overlapAdd(stft_matrix, fft_size, hop_size)
%OVERLAPADD Perform overlap-add ISTFT reconstruction
%
% Optimized for edge devices with minimal memory allocation

    [num_freqs, num_frames] = size(stft_matrix);

    % Estimate output length
    output_length = (num_frames - 1) * hop_size + fft_size;
    output_signal = zeros(output_length, 1, 'single');
    window_sum = zeros(output_length, 1, 'single');

    % Synthesis window
    window = single(hann(fft_size, 'periodic'));

    for frame_idx = 1:num_frames
        % Reconstruct frequency domain signal (full FFT bins)
        freq_signal = zeros(fft_size, 1, 'single');
        freq_signal(1:num_freqs) = stft_matrix(:, frame_idx);

        % Conjugate symmetry for real signal
        if num_freqs < fft_size
            freq_signal(num_freqs+1:end) = conj(flipud(freq_signal(2:fft_size-num_freqs+2)));
        end

        % Inverse FFT
        time_signal = real(ifft(freq_signal, fft_size));

        % Apply synthesis window
        time_signal = time_signal .* window;

        % Overlap-add
        start_idx = (frame_idx - 1) * hop_size + 1;
        end_idx = start_idx + fft_size - 1;

        output_signal(start_idx:end_idx) = output_signal(start_idx:end_idx) + time_signal;
        window_sum(start_idx:end_idx) = window_sum(start_idx:end_idx) + (window.^2);
    end

    % Normalize by window sum
    nonzero_idx = window_sum > eps('single');
    output_signal(nonzero_idx) = output_signal(nonzero_idx) ./ window_sum(nonzero_idx);
end
