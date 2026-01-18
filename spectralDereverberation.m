function dereverb_signal = spectralDereverberation(reverb_signal, fs)
%SPECTRALDEREVERBERATION Spectral subtraction-based dereverberation for edge devices
%
% Implements a lightweight dereverberation algorithm suitable for
% resource-constrained edge devices. Uses spectral subtraction to
% reduce late reverberation.
%
% Inputs:
%   reverb_signal - Reverberant signal [samples x 1]
%   fs            - Sampling frequency (Hz)
%
% Output:
%   dereverb_signal - Dereverberated signal [samples x 1]
%
% Edge Optimizations:
%   - Single precision arithmetic
%   - Efficient short-time processing
%   - Minimal memory footprint
%   - MATLAB Coder compatible

%% Type conversion
reverb_signal = single(reverb_signal(:));
fs = single(fs);

%% Parameters
frame_length = int32(512);     % Frame size (~32 ms at 16 kHz)
hop_size = frame_length / 2;   % 50% overlap
num_samples = length(reverb_signal);

% Spectral subtraction parameters
alpha = single(0.95);          % Over-subtraction factor
beta = single(0.01);           % Spectral floor (prevent negative values)

%% Estimate late reverberation spectrum
% Use the reverberant tail of the signal to estimate reverberation characteristics
% Take the last 10% of signal as noise/reverberation estimate
tail_start = round(0.9 * num_samples);
reverb_tail = reverb_signal(tail_start:end);

% Compute average spectrum of reverberant tail
nfft = frame_length;
num_tail_frames = floor(length(reverb_tail) / hop_size) - 1;
reverb_spectrum_sum = zeros(nfft/2 + 1, 1, 'single');

window = single(hann(frame_length, 'periodic'));

for i = 1:num_tail_frames
    start_idx = (i - 1) * hop_size + 1;
    end_idx = start_idx + frame_length - 1;

    if end_idx <= length(reverb_tail)
        frame = reverb_tail(start_idx:end_idx) .* window;
        fft_frame = fft(frame, nfft);
        reverb_spectrum = abs(fft_frame(1:nfft/2 + 1));
        reverb_spectrum_sum = reverb_spectrum_sum + reverb_spectrum.^2;
    end
end

reverb_spectrum_avg = sqrt(reverb_spectrum_sum / num_tail_frames);

%% STFT Processing with spectral subtraction
num_frames = floor((num_samples - frame_length) / hop_size) + 1;
dereverb_stft = zeros(nfft/2 + 1, num_frames, 'single');

for frame_idx = 1:num_frames
    % Extract frame
    start_idx = (frame_idx - 1) * hop_size + 1;
    end_idx = start_idx + frame_length - 1;

    if end_idx > num_samples
        break;
    end

    % Apply window and compute STFT
    frame = reverb_signal(start_idx:end_idx) .* window;
    fft_frame = fft(frame, nfft);
    magnitude = abs(fft_frame(1:nfft/2 + 1));
    phase = angle(fft_frame(1:nfft/2 + 1));

    % Spectral subtraction
    % Enhanced_magnitude = max(Original_magnitude - alpha * Reverb_magnitude, beta * Original_magnitude)
    enhanced_magnitude = max(magnitude - alpha * reverb_spectrum_avg, ...
                            beta * magnitude);

    % Reconstruct with original phase
    dereverb_stft(:, frame_idx) = enhanced_magnitude .* exp(1j * phase);
end

%% Inverse STFT with overlap-add
dereverb_signal = overlapAddDereverb(dereverb_stft, frame_length, hop_size, num_samples);

% Normalize output
dereverb_signal = dereverb_signal / (max(abs(dereverb_signal)) + eps('single'));

end

function output_signal = overlapAddDereverb(stft_matrix, frame_length, hop_size, target_length)
%OVERLAPADDDEREVERB Overlap-add ISTFT for dereverberation
%
% Optimized for edge devices

    [num_freqs, num_frames] = size(stft_matrix);
    nfft = (num_freqs - 1) * 2;

    % Estimate output length
    output_length = (num_frames - 1) * hop_size + frame_length;
    output_signal = zeros(output_length, 1, 'single');
    window_sum = zeros(output_length, 1, 'single');

    % Synthesis window
    window = single(hann(frame_length, 'periodic'));

    for frame_idx = 1:num_frames
        % Reconstruct full spectrum with conjugate symmetry
        freq_signal = zeros(nfft, 1, 'single');
        freq_signal(1:num_freqs) = stft_matrix(:, frame_idx);
        freq_signal(num_freqs+1:end) = conj(flipud(freq_signal(2:nfft-num_freqs+2)));

        % Inverse FFT
        time_signal = real(ifft(freq_signal, nfft));

        % Apply synthesis window
        time_signal = time_signal .* window;

        % Overlap-add
        start_idx = (frame_idx - 1) * hop_size + 1;
        end_idx = start_idx + frame_length - 1;

        output_signal(start_idx:end_idx) = output_signal(start_idx:end_idx) + time_signal;
        window_sum(start_idx:end_idx) = window_sum(start_idx:end_idx) + (window.^2);
    end

    % Normalize by window sum
    nonzero_idx = window_sum > eps('single');
    output_signal(nonzero_idx) = output_signal(nonzero_idx) ./ window_sum(nonzero_idx);

    % Trim to target length
    output_signal = output_signal(1:min(target_length, length(output_signal)));
end
