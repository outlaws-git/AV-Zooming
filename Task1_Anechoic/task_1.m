%[text] # IEEE Signal Processing Cup 2026 on AV Zoom - Task 1
%[text] 
%[text:tableOfContents]{"heading":"Table of Contents"}
clc; clear; close all;
%%
%[text] ## Simulation Setup
%[text] ### Parameters Setup
%[text] Set simulation scenario parameters such as sampling rate, sound speed and etc.
fs = 16e3;                                 % sampling rate = 16 kHz
c = 340;                                   % speed of sound in air, approximately 340 m/s at 20 °C

desired_sir = 0;                           % desired Signal-to-Interference Ratio (SIR)
desired_snr = 5;                           % desired Signal-to-Noise Ratio (SNR)

num_sources = 2;                           % number of sources 
target_id = 1;                             % id of target source
interf_id = 2;                             % id of interference source
%[text] Set other parameters.
t_length_play = 10;                        % time duration for playing an audio
t_play = 1/fs:1/fs:t_length_play;          % time sequence in the duration
num_play_samples = t_length_play*fs;       % number of samples in the duration
%%
%[text] ### **Room Setup**
%[text] Define an shoebox room with the dimensions (width, length and height, respectively) in meters. 
room_dim = [4.9 4.9 4.9];                  % the room's width, length and height in meters
room_surf_num = 6;                         % the shoebox room has 6 surfaces (floor, front, back, left, right, and ceiling)
%%
%[text] ### **Microphone Array Setup**
%[text] Define a 2-microphone Uniform Linear Array \[1\]. 
mic_positions = [2.41 2.45 1.5; ...
                 2.49 2.45 1.5];
num_mics = size(mic_positions,1);
mic_center = mean(mic_positions,1);
mic_distance = norm(mic_positions(1,:) - mic_positions(2,:));   % distance between the microphones: 8cm

microphone = ...
    phased.OmnidirectionalMicrophoneElement('FrequencyRange',[20 8000]);
micULA = phased.ULA(num_mics,mic_distance,'Element',microphone);
%[text] Define a wideband collector converts incident wideband wave fields arriving from specified directions into signals to be further processed \[1\].
%[text] Experiments to Conduct:
%[text] - Add  'NumSubbands', 1000, ... since the default value is 1. More subbands -\> more computationally expensive but Accurate
%[text] -  \
collector = phased.WidebandCollector('Sensor', micULA, ...
    'PropagationSpeed', c, ...
    'SampleRate', fs, ...
    'ModulatedInput', false);
%%
%[text] ### **Audio Sources Setup**
%[text] %[text:anchor:TMP_554a] **Target Source: Source 1 (Target Speaker – Male Speech)**
%[text] - Azimuth: 90° (directly in front of array)
%[text] - Height: 1.5 m
%[text] - Position: (2.45, 3.45, 1.5) m
%[text] - Dataset: Male speech from MUSAN Corpus ([OpenSLR 17](https://www.openslr.org/17)) \
% target_file = [pwd '\..\musan\speech\librivox\speech-librivox-0000.wav'];  % a male speech in MUSAN
% This builds the path correctly regardless of your OS
target_file = fullfile(pwd,  'musan', 'speech', 'librivox', 'speech-librivox-0000.wav');
target_signal = audioread(target_file);
target_signal = target_signal./max(abs(target_signal));

target_positions = [2.45 3.45 1.5];
target_angles = [90 0];                     % azimuth = 90 degree, elevation = 0 degree
soundsc(target_signal(1:num_play_samples),fs)
%[text] Verify the azimuth and elevation angels of the target source.
v = target_positions - mic_center; 
[az_rad, el_rad, ~] = cart2sph(v(:,1), v(:,2), v(:,3));
target_azimuth = rad2deg(az_rad) %[output:4f34d9b8]
target_elevation = rad2deg(el_rad) %[output:88c31395]
%%
%[text] %[text:anchor:TMP_1c7f] **Interference Source: Source 2 (Interference)**
%[text] - Azimuth: 40° (to the right-front)
%[text] - Height: 1.5 m
%[text] - Position: (3.22, 3.06, 1.5) m
%[text] - Dataset: Female speech, Babble, traffic/car, café, office fan from MUSAN Corpus (OpenSLR 17) \
interf_file =fullfile(pwd,  'musan', 'speech', 'librivox', 'speech-librivox-0100.wav');   % a female speech in MUSAN

interf_signal = audioread(interf_file);
interf_signal = interf_signal./max(abs(interf_signal));

interf_positions = [3.22 3.06 1.5];
interf_angles = [40 0];                     % azimuth = 40 degree, elevation = 0 degree
soundsc(interf_signal(1:num_play_samples),fs)
%[text] Verify the azimuth and elevation angels of the interference source.
v = interf_positions - mic_center; 
[az_rad, el_rad, ~] = cart2sph(v(:,1), v(:,2), v(:,3));
interf_azimuth = rad2deg(az_rad) %[output:7bd987d2]
interf_elevation = rad2deg(el_rad) %[output:933c7c63]
%%
%[text] %[text:anchor:TMP_0237] **Align Signal Lengths**
%[text] Align the target and interference signal lengths using one of two methods: 
%[text] - Method 0: Pad the shorter signal with zeros to match the length of the longer signal.
%[text] - Method 1: Truncate the longer signal to match the length of the shorter signal. \
method = 0;                                 
if method == 0
    signal_length = max(numel(target_signal),numel(interf_signal));
    target_signal = [target_signal; zeros(signal_length-numel(target_signal),1)];
    interf_signal = [interf_signal; zeros(signal_length-numel(interf_signal),1)];
elseif method == 1
    signal_length = min(numel(target_signal),numel(interf_signal));
    target_signal = target_signal(1:signal_length);
    interf_signal = interf_signal(1:signal_length);
end
%%
%[text] %[text:anchor:TMP_2ed6] **Room and Signal Visulization**
%[text] Plot the room space along with the target source (green circle), the interference source (red circle) and the receiving micophones (blue x).
h = figure;
plotRoom(room_dim,mic_positions,[target_positions;interf_positions],h) %[output:851559af]
%[text] Plot the original signal segments.
figure %[output:1b5a5db1]
tiledlayout(2,1) %[output:1b5a5db1]

nexttile %[output:1b5a5db1]
plot(t_play,target_signal(1:(num_play_samples))) %[output:1b5a5db1]
title("Original Target Signal (Male Speech)") %[output:1b5a5db1]
grid on %[output:1b5a5db1]

nexttile %[output:1b5a5db1]
plot(t_play,interf_signal(1:(num_play_samples))) %[output:1b5a5db1]
title("Original Interference Signal (Female Speech)") %[output:1b5a5db1]
grid on %[output:1b5a5db1]
xlabel("Time (s)") %[output:1b5a5db1]

%%
%[text] ## Received Signal Generation
%[text] ### SIR Adjustment
%[text] Tune the interference signal so it meets the desired Signal-to-Interference Ratio (SIR) =  0 dB (equal power target & interferer).
target_norm = norm(target_signal);
goal_interf_norm = target_norm/(10^(desired_sir/20));
factor = goal_interf_norm./norm(interf_signal);

interf_signal = interf_signal.*factor;
preRIR_sirCheck_0dB = 10*log10(mean(target_signal.^2)/mean(interf_signal.^2)) %[output:5601ebfe]
%%
%[text] Package the source signals, positions and angles into a metrics.
source_signals = [target_signal';interf_signal'];
source_positions = [target_positions; interf_positions];
source_angles = [target_angles; interf_angles];
%%
%[text] ### Generate Room Impulse Response
%[text] In MATLAB, we can use the [`acousticRoomResponse`](https://www.mathworks.com/help/audio/ref/acousticroomresponse.html) to generate Room Impulse Response \[3-4\].
%[text] Define octave bands for Room Impulse Response generation.
band_center_freq = [125,250,500,1000,...   % center frequency of 7 octave bands
    2000,4000,8000];
num_bands = numel(band_center_freq);       % number of bands
%[text] RT60 (Reverberation Time) is a fundamental parameter in room acoustics that quantifies how long sound persists in a space after the sound source stops. It is defined as the time required for the sound pressure level (SPL) to decay by 60 decibels from its original value after the sound source ceases.
%[text] **Anechoic Chamber** 
%[text] Reverberation Time (RT60): ≈ 0.0 s -\> absorption coefficient = 1 (full absorption), scattering = 0 (specular, no scattering). 
%[text] The shoebox room has 6 surfaces while the number of frequency bands must match band\_center\_freq, here 7 octave bands.
absWall_anechoic = ones(room_surf_num,num_bands);      % material absorption coefficients of walls
scatWall_anechoic = zeros(room_surf_num,num_bands);    % material scattering coefficients
%[text] %[text:anchor:TMP_429d] Generate Room Impulse Response for Task 1: Anechoic Chamber.
rir_anechoic = cell(num_sources,1);
for source_id = 1:num_sources
    rir_anechoic{source_id} = acousticRoomResponse(room_dim, ...  % the dimensions of a shoebox room or a triangulation object from an STL file using stlread used to describe a room
        source_positions(source_id,:), ...                        % cartesian coordinates of the transmitter in meters
        mic_positions,...                                         % cartesian coordinates of the receiver in meters
        SampleRate= fs, ...                                       % sample rate of the impulse response in Hertz
        SoundSpeed= c, ...                                        % speed of sound in meters per second
        Algorithm="hybrid",...                                    % impulse response synthesis algorithm: "hybrid" (defult), "image-source", "stochastic ray tracing"
        ImageSourceOrder=0,...                                    % image-source maximum order, 0 = no image source
        NumStochasticRays=0,...                                   % number of rays to use for the stochastic ray tracing method, 0 = no rays
        MaxNumRayReflections=0,...                                % maximum number of reflections per stochastic ray
        ReceiverRadius= 0.5,...                                   % maximum number of reflections per stochastic ray
        AirAbsorption=0,...                                       % air absorption coefficient, 0 = no air absorption
        MaterialAbsorption= absWall_anechoic,...                  % material absorption coefficients
        MaterialScattering= scatWall_anechoic,...                 % material scattering coefficients
        BandCenterFrequencies= band_center_freq);                 % center frequencies of the bandpass filters in Hertz
end

%%
%[text] ### Generate Received Signals
%[text] **Anechoic Chamber** 
%[text] Simulate the received audio by filtering with the source signals with room impulse response.
anechoic_rcv_clean = zeros(num_sources,num_mics,signal_length);
for source_id = 1:num_sources
    for mic_id = 1:num_mics
        anechoic_rcv_clean(source_id,mic_id,:) = filter(rir_anechoic{source_id}(mic_id,:),1,source_signals(source_id,:));
    end
end
%[text] Use a **wideband collector** \[**Not used in the demo, Just for your reference**\] to simulate the signal received by the array. Notice that this approach assumes that each input single-channel signal is received at the origin of the array by a single microphone .
anechoic_rcv_clean = zeros(num_sources,num_mics,signal_length);
for source_id = 1:num_sources
    anechoic_rcv_clean(source_id,:,:) = ...
        reshape(collector(source_signals(source_id,:)',source_angles(source_id,:)')',1,num_mics,signal_length);
end
%%
%[text] ### SNR Adjustment
%[text] **Anechoic Chamber**
%[text] Check Signal-to-Interference Ratio (SIR) =  0 dB (equal power target & interferer).
anechoic_postRIR_SirCheck_0dB = 10*log10(mean((anechoic_rcv_clean(target_id,:,:).^2),3)./mean((anechoic_rcv_clean(target_id,:,:).^2),3)) %[output:5a4f8c86]
%[text] Add the received target and interference signals at the microphones.
anechoic_rcv_SI = reshape(anechoic_rcv_clean(target_id,:,:) + anechoic_rcv_clean(interf_id,:,:),num_mics,[]);
%[text] Add noise so the Signal-to-Noise Ratio (SNR) = 5 dB (additive background noise).
anechoic_rcv_noisy = awgn(anechoic_rcv_SI,desired_snr,'measured'); % randn or dsp.ColoredNoise can be used
anechoic_snrCheck_5dB = 10*log10(mean((anechoic_rcv_SI.^2),2)./mean(((anechoic_rcv_noisy-anechoic_rcv_SI).^2),2)) %[output:6197802b]
%%
%[text] **Play an segment of the received signals** 
%[text] Anechoic Chamber
soundsc(anechoic_rcv_noisy(:,1:num_play_samples)',fs)
%%
%[text] Save (saveOrLoad = 0) or load (saveOrLoad = 1) the workspace with all the variables for (futher) testing.
clc; clear; close all;
saveOrLoad = 1; 

%Create a timestamp string in the format YYYYMMDD_HHMMSS
timestamp = datestr(datetime('now'), 'yyyymmdd_HHMMSS'); 
filename = ['my_workspace_' timestamp '.mat'];  % Construct filename
filename = 'my_workspace_20260115_181543.mat';

if ~saveOrLoad
   save(filename);
else
   load(filename);
end
%%
%[text] ## Related Techniques
%[text] This section demonstrates how to implement some relevant techniques in MATLAB, such as beamforming \[1-2\], direction of arrival estimation \[5\], and speaker separation \[6-7\]. For speech denoising using deep learning networks, please refer to \[8\].
%%
%[text] ### Direction of Arrival Estimation -[Notebook lm](https://notebooklm.google.com/notebook/f519718b-e4db-4716-af42-78c925643d0c)
%[text] Use direction-of-arrival (DOA) estimation to localize the direction of a radiating or reflecting source \[5\].
%[text] **GCC Estimator**
%[text] The [`phased.GCCEstimator`](https://www.mathworks.com/help/phased/ref/phased.gccestimator-system-object.html) System object™ creates a direction of arrival estimator for wideband signals. This System object estimates the direction of arrival or time of arrival among sensor array elements using the generalized cross-correlation with phase transform algorithm (GCC-PHAT) . You can estimate the broadside arrival angle ([Broadside Angles](https://www.mathworks.com/help/phased/ug/spherical-coordinates.html)) of the plane wave with respect to the line connecting the two microphones. 
gcc_estimator = phased.GCCEstimator('SensorArray',micULA,...
    'PropagationSpeed',c,'SampleRate',fs);
%[text] Use the `GCCEstimator to estimate the` direction-of-arrival in two Tasks.
anechoic_combined_doa = gcc_estimator(anechoic_rcv_noisy')
%%
%[text] ### Beamforming
%[text] Beamformers enhance detection of signals by coherently summing signals across elements of arrays. Conventional beamformers have fixed weights while adaptive beamformers have weights that respond to the environment. Use adaptive beamformers to reject spurious or interfering signals from non-target directions. The Phased Array System toolbox supports narrowband and wideband beamformers \[2\].
%[text] **Time Delay Beamformer**
%[text] The [`phased.TimeDelayBeamformer`](https://www.mathworks.com/help/phased/ref/phased.timedelaybeamformer-system-object.html) System object™ object implements a time delay beamformer. The object performs delay and sum beamforming on the received signal by using time delays.
td_beamformer = phased.TimeDelayBeamformer('SensorArray',micULA,...
    'PropagationSpeed',c,'SampleRate',fs,'Direction',target_angles');

td_bf_anechoic = td_beamformer(anechoic_rcv_noisy');
soundsc(td_bf_anechoic(1:num_play_samples)',fs)
%%
%% 1. Setup Parameters
fs = 44100;                 % Sampling frequency
c = 343;                    % Speed of sound
f_signal = 1000;            % Frequency of our test tone
t = (0:1/fs:0.5)';          % 0.5 seconds of audio
target_angle = 30;          % Where the "voice" is coming from (Degrees)

% Define the Array (2 Microphones, 5cm apart)
mic_spacing = 0.05; 
micULA = phased.ULA('NumElements', 2, 'ElementSpacing', mic_spacing);

%% 2. Create the Noisy Signal
% Create a steering vector for the target angle
steerVec = phased.SteeringVector('SensorArray', micULA, 'PropagationSpeed', c);
sv = steerVec(f_signal, target_angle);

% Simulate the signal hitting 2 mics with a slight delay/phase difference
clean_signal = sin(2*pi*f_signal*t);
micsig = clean_signal * sv'; % This creates a [Samples x 2] matrix

% Add some random noise to each microphone
micsig_noisy = micsig + 0.5*randn(size(micsig));

%% 3. Apply Time Delay Beamforming
% Initialize the beamformer
td_beamformer = phased.TimeDelayBeamformer('SensorArray', micULA, ...
    'PropagationSpeed', c, 'SampleRate', fs, 'Direction', [target_angle; 0]);

% Run the beamformer
% Input: [Samples x Channels] -> Output: [Samples x 1]
beamformed_out = td_beamformer(micsig_noisy);

%% 4. Visualization (The "Workable" Part)
figure('Color', 'w', 'Position', [100, 100, 800, 600]);

% Plot 1: The Input vs Output signals
subplot(2,1,1);
plot(t, micsig_noisy(:,1), 'r', 'DisplayName', 'Mic 1 (Noisy)'); hold on;
plot(t, beamformed_out, 'b', 'LineWidth', 1.5, 'DisplayName', 'Beamformed Output');
xlim([0 0.01]); % Zoom in to see the waveforms
title('Time Domain: Noisy Input vs. Cleaned Output');
xlabel('Time (s)'); ylabel('Amplitude');
legend; grid on;

% Plot 2: The "Beam" (Spatial Response)
% This shows where the microphones are "listening"
subplot(2,1,2);
pattern(micULA, f_signal, -180:180, 0, 'PropagationSpeed', c, ...
    'Type', 'directivity', 'CoordinateSystem', 'polar');
title(['Spatial Response (Pointed at ', num24str(target_angle), '°)']);
%%
%[text] ### Speaker Seperation
%[text] #### Load a Pretrained Model
%[text] Execute the following commands to download and unzip a pretrained speaker separation model [`separateSpeakers`](https://ww2.mathworks.cn/help/audio/ref/separatespeakers.html) to your temporary directory.   
%downloadFolder = fullfile(tempdir,"separateSpeakerDownload");
%loc = websave(downloadFolder,"https://ssd.mathworks.com/supportfiles/audio/separateSpeakers.zip");
%modelsLocation = tempdir;
%unzip(loc,modelsLocation)
%addpath(fullfile(modelsLocation,"separateSpeakers"))
%addpath(fullfile(pwd,"separateSpeakers"))
%%
addpath(fullfile("separateSpeakers"))
addpath(fullfile(pwd,"separateSpeakers"))
%%
%[text] #### **Method 1: Use separateSpeakers and then Combine Signals** [**NotebookLM**](https://notebooklm.google.com/notebook/49228b36-4f58-46ab-97f5-b83e99f2115f)
%[text] **Anechoic Chamber**
%[text] Use the pre-trained speaker separation model to seperate target and interference signals at the two microphone elements respectively:
anechoic_mic1_out = separateSpeakersTest(anechoic_rcv_noisy(1,:),signal_length,fs); %[text:anchor:TMP_6230]
anechoic_mic2_out = separateSpeakersTest(anechoic_rcv_noisy(2,:),signal_length,fs);
%[text] Calulate the average signal:
anechoic_avg_out = (anechoic_mic1_out + anechoic_mic2_out)/2;
%%
false %[control:checkbox:8b99]{"position":[1,6]}
soundsc(anechoic_avg_out(1:num_play_samples,target_id)',fs)
%soundsc(anechoic_avg_out(1:num_play_samples,interf_id)',fs)
%[text] Estimate the DoA: the broadside-reference arrival angle ([Broadside Angles](https://www.mathworks.com/help/phased/ug/spherical-coordinates.html)) measured from the array normal.
%[text] ![](text:image:519b)
anechoic_target_doa = gcc_estimator([anechoic_mic1_out(:,target_id) anechoic_mic2_out(:,target_id)]) % true value: 0 degree
anechoic_interf_doa = gcc_estimator([anechoic_mic1_out(:,interf_id) anechoic_mic2_out(:,interf_id)]) % true value: 50 degree
%%
%[text] **Reverberant Room** 
%[text] Use the pre-trained speaker separation model to seperate target and interference signals at the two microphone elements respectively:
% reverb_mic1_out = separateSpeakersTest(reverb_rcv_noisy(1,:),signal_length,fs); %[text:anchor:TMP_6230]
% reverb_mic2_out = separateSpeakersTest(reverb_rcv_noisy(2,:),signal_length,fs);
%[text] Calulate the average signal:
% reverb_avg_out = (reverb_mic1_out + reverb_mic2_out)/2;
% soundsc(reverb_avg_out(1:num_play_samples,target_id)',fs)
% soundsc(reverb_avg_out(1:num_play_samples,interf_id)',fs)
%[text] Estimate the DoA: the broadside-reference arrival angle ([Broadside Angles](https://www.mathworks.com/help/phased/ug/spherical-coordinates.html)) measured from the array normal.
% reverb_target_doa = gcc_estimator([reverb_mic1_out(:,target_id) reverb_mic2_out(:,target_id)]) % true value: 0 degree
% reverb_interf_doa = gcc_estimator([reverb_mic1_out(:,interf_id) reverb_mic2_out(:,interf_id)]) % true value: 50 degree
%%
%[text] #### **Method 2: Combining Signals and then Use separateSpeakers**
%[text] Combine the received signals across the 2-microphone elements of the array.
%[text] **Use mean value**
anechoic_rcv_mean = mean(anechoic_rcv_noisy,1);
reverb_rcv_mean = mean(anechoic_rcv_noisy,1);

target_anechoic_mean = separateSpeakersTest(anechoic_rcv_mean,signal_length,fs);
soundsc(target_anechoic_mean(1:num_play_samples,target_id)',fs)
%soundsc(target_anechoic_mean(1:num_play_samples,interf_id)',fs)
%%
% target_reverb_mean = separateSpeakersTest(reverb_rcv_mean,signal_length,fs);
% soundsc(target_reverb_mean(1:num_play_samples,target_id)',fs)
% soundsc(target_reverb_mean(1:num_play_samples,interf_id)',fs)

%%
%[text] **Use Time Delay Beamforming**
target_anechoic_TDBF = separateSpeakersTest(td_bf_anechoic',signal_length,fs);
%target_reverb_TDBF = separateSpeakersTest(td_bf_reverb',signal_length,fs);
%%
%[text] ## Metics Calculation
%[text] This section show how to calculate the required metrics in different cases.
%[text] %[text:anchor:TMP_4915] **Anechoic Chamber**
%[text] Method1: Use the pre-trained speaker separation model to remove the interference at the two microphone elements respectively:
merics_anechoic_method1 = calcMetrics(target_signal,anechoic_avg_out(:,target_id),fs)
%[text] Method 2: Combining Microphone Array Signals and then separateSpeakers
merics_anechoic_method2_mean = calcMetrics(target_signal,target_anechoic_mean(:,target_id),fs)
merics_anechoic_method2_TDBF = calcMetrics(target_signal,target_anechoic_TDBF(:,target_id),fs)
%[text] **Reverberant Room** 
%[text] Method1: Use the pre-trained speaker separation model to remove the interference at the two microphone elements respectively:
merics_reverb_method1 = calcMetrics(target_signal,reverb_avg_out(:,target_id),fs)
%[text] Method 2: Combining Microphone Array Signals and then separateSpeakers
merics_reverb_method2_mean = calcMetrics(target_signal,target_reverb_mean(:,target_id),fs)
merics_reverb_method2_TDBF = calcMetrics(target_signal,target_reverb_TDBF(:,target_id),fs)
%%
%[text] ## Submission Files Generation
%[text] Create submission folders.
task1_folder_name = fullfile(pwd, 'Task1_Anechoic');

if ~exist(task1_folder_name, 'dir')
    mkdir(task1_folder_name);
end

task2_folder_name = fullfile(pwd, 'Task2_Reverberant');
if ~exist(task2_folder_name, 'dir')
    mkdir(task2_folder_name);
end
%[text] Collect common simulation parameters. 
params.sampling_rate = fs;
params.sound_speed = c;
params.room_dimensions = room_dim;
params.microphone_positions = mic_positions;
params.array_spacing = mic_distance;

params.target_position = target_positions;
params.target_azimuth = target_angles(1);
params.target_hight = target_positions(3);

params.interference_position = interf_positions;
params.interference_azimuth = interf_angles(1);
params.interence_hight = interf_positions(3);

params.sir = desired_sir;
params.snr = desired_snr;
%[text] **Anechoic Chamber**
%[text] Save signals and paramters for Task 1: Anechoic Chamber (the Time Delay Beamforming case).
interference_signal = interf_signal;
mixture_signal = anechoic_rcv_noisy';
rir_data = rir_anechoic;
processed_signal = target_anechoic_mean(:,target_id);
metrics = merics_anechoic_method2_mean;

save(fullfile(task1_folder_name,['Task1_Anechoic_SNR' num2str(desired_snr) 'db.mat']), ...
    'target_signal','interference_signal','mixture_signal',...
    'processed_signal','rir_data','params','metrics');

audiowrite(fullfile(task1_folder_name,'target_signal.wav'), target_signal, fs);
audiowrite(fullfile(task1_folder_name,'interference_signal1.wav'), interf_signal, fs);
%[text] **Reverberant Room**
%[text] Save signals and paramters for Task 2: Reverberant Room (the no Beamforming case).
interference_signal = interf_signal;
mixture_signal = reverb_rcv_noisy';
rir_data = rir_reverb;
processed_signal = target_reverb_mean(:,target_id);
metrics = merics_reverb_method2_mean;

save(fullfile(task2_folder_name,['Task2_Reverberant_SNR' num2str(desired_snr) 'db.mat']), ...
    'target_signal','interference_signal','mixture_signal',...
    'processed_signal','rir_data','params','metrics');

audiowrite(fullfile(task2_folder_name,'target_signal.wav'), target_signal, fs);
audiowrite(fullfile(task2_folder_name,'interference_signal1.wav'), interf_signal, fs);
%%
%[text] ## Helper Functions
%[text] **plotRoom**
%[text] This helper function is used to plot 3D room with receiver/transmitter points \[RIR\_SRT\].  
function plotRoom(roomDimensions,rx,tx,figHandle)
figure(figHandle)
X = [0;roomDimensions(1);roomDimensions(1);0;0];
Y = [0;0;roomDimensions(2);roomDimensions(2);0];
Z = [0;0;0;0;0];
figure;
hold on;
plot3(X,Y,Z,"k",LineWidth=1.5);
plot3(X,Y,Z+roomDimensions(3),"k",LineWidth=1.5);
set(gca,"View",[-28,35]);
for k=1:length(X)-1
    plot3([X(k);X(k)],[Y(k);Y(k)],[0;roomDimensions(3)],"k",LineWidth=1.5);
end
grid on
xlabel("X (m)")
ylabel("Y (m)")
zlabel("Z (m)")

plot3(tx(1,1),tx(1,2),tx(1,3),"go",LineWidth=2)
plot3(tx(2,1),tx(2,2),tx(2,3),"ro",LineWidth=2)
num_rx = size(rx,1);
for rx_id = 1:num_rx
    plot3(rx(rx_id,1),rx(rx_id,2),rx(rx_id,3),"bx",LineWidth=1)
end
end
%%
%[text] **separateSpeakersTest**
%[text] This helper function is used to test speaker separation using a pretrained Model \[5-6\].
function speaker_separated = separateSpeakersTest(rcv_signal,signal_length,fs)
num_speakers = 2;

t_length_frame = 3;                                   % time duration for processing an audio signal
num_samples_per_frame = t_length_frame*fs;            % number of samples in the duration
num_frames = ceil(signal_length/num_samples_per_frame);

extended_signal_length = max(signal_length,num_frames*num_samples_per_frame);
extended_signal = [rcv_signal zeros(1,extended_signal_length-signal_length)];

% audioWriter = audioDeviceWriter('SampleRate',fs, ...
%     'SupportVariableSizeInput', true);
% isAudioSupported = (length(getAudioDevices(audioWriter))>1);
speaker_separated_extend = zeros(extended_signal_length,num_speakers);

for n = 1:num_frames
    idx_start = (n-1)*num_samples_per_frame + 1;
    idx_end = idx_start + num_samples_per_frame - 1;
%[text] Call `separateSpeakers.`Number of speakers to separate, specified as `1`, `2`, or `3`. If you do not specify `NumSpeakers`, `separateSpeakers` estimates the number of speakers. You can specify an additional output argument `r` to obtain the residual. For more information, see [One-And-Rest Speech Separation](https://www.mathworks.com/help/audio/ref/separatespeakers.html#mw_7558f471-0e7c-4068-8d37-696e6e6cd3fc_head).
    [y,r] = separateSpeakers(extended_signal(idx_start:idx_end),fs);

    % if isAudioSupported
    %     play(audioWriter,y(:,1));
    % end

    speaker_separated_extend(idx_start:idx_end,1) = y(:,1);
    if size(y,2) > 1
        speaker_separated_extend(idx_start:idx_end,2) = y(:,2);
    end
end
speaker_separated = speaker_separated_extend(1:signal_length,:);
% release(audioWriter)
end
%%
%[text] **calcMetrics**
%[text] This helper function is used to calculate the required metrics.
function merics = calcMetrics(clean_signal,enhanced_signal,fs)
% merics.osinr = 10*log10(mean(clean_signal.^2)/(mean(enhanced_signal-clean_signal).^2));
%[text] Use [`sisnr`](https://www.mathworks.com/help/audio/ref/sisnr.html) to calculate the Scale-invariant signal-to-noise ratio
merics.sisnr = sisnr(enhanced_signal,clean_signal);
%[text] Use [`stoi`](https://www.mathworks.com/support/search.html) to calculate the Short-time objective intelligibility measure \[9\]
merics.stoi = stoi(enhanced_signal,clean_signal,fs);
%[text] Use [`visqol`](https://www.mathworks.com/support/search.html) to calculate the Objective metric for perceived audio quality \[9\]
%[text] - `"MOS"` — The output is a scalar representing the mean opinion score (MOS) in the range \[1,5\], where a higher value corresponds to higher quality.
%[text] - `"NSIM"` — The output is a scalar representing the neurogram similarity index measure (NSIM) in the range \[-1,1\], where 1 corresponds to a perfect similarity between the degraded and reference signals. In practice, the NSIM is generally in the range \[0,1\]. \
merics.visqol = visqol(enhanced_signal,clean_signal,fs,Mode="speech",OutputMetric="MOS and NSIM");
end
%%
%[text] ## References
%[text] \[1\] Acoustic Beamforming Using a Microphone Array: [https://www.mathworks.com/help/audio/ug/acoustic-beamforming-using-a-microphone-array.html](https://ww2.mathworks.cn/help/audio/ug/acoustic-beamforming-using-a-microphone-array.html) 
%[text] \[2\] Beamforming: [https://ww2.mathworks.cn/help/phased/beamforming.html](https://ww2.mathworks.cn/help/phased/beamforming.html) 
%[text] %[text:anchor:TMP_2971] \[3\] Room Impulse Response Simulation with Image Source Method and HRTF Interpolation: [https://www.mathworks.com/help/audio/ug/room-impulse-response-simulation-with-image-source-method-and-hrtf-interpolation.html](https://www.mathworks.com/help/audio/ug/room-impulse-response-simulation-with-image-source-method-and-hrtf-interpolation.html)
%[text] \[4\] Room Impulse Response Simulation with Stochastic Ray Tracing: [https://www.mathworks.com/help/audio/ug/room-impulse-response-simulation-with-stochastic-ray-tracing.html](https://www.mathworks.com/help/audio/ug/room-impulse-response-simulation-with-stochastic-ray-tracing.html)
%[text] \[5\] Direction of Arrival Estimation: [https://ww2.mathworks.cn/help/phased/direction-of-arrival-doa-estimation-1.html](https://ww2.mathworks.cn/help/phased/direction-of-arrival-doa-estimation-1.html)
%[text] \[6\] Compare Speaker Separation Models: [https://www.mathworks.com/help/audio/ug/compare-speaker-separation-models.html](https://www.mathworks.com/help/audio/ug/compare-speaker-separation-models.html) 
%[text] \[7\] Train End-to-End Speaker Separation Model: [https://www.mathworks.com/help/audio/ug/end-to-end-deep-speech-separation.html](https://www.mathworks.com/help/audio/ug/end-to-end-deep-speech-separation.html) 
%[text] \[8\] Denoise Speech Using Deep Learning Networks: [https://ww2.mathworks.cn/help/releases/R2025b/audio/ug/denoise-speech-using-deep-learning-networks.html](https://ww2.mathworks.cn/help/releases/R2025b/audio/ug/denoise-speech-using-deep-learning-networks.html) 
%[text] \[9\] Measuring Speech Intelligibility and Perceived Audio Quality with STOI and ViSQOL: [https://www.mathworks.com/help/audio/ug/measure-speech-intelligibility-and-perceived-audio-quality-with-stoi-and-visqol.html](https://www.mathworks.com/help/audio/ug/measure-speech-intelligibility-and-perceived-audio-quality-with-stoi-and-visqol.html) 
%[text] 
%[text]  Copyright 2025 The MathWorks, Inc.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":18}
%---
%[control:checkbox:8b99]
%   data: {"defaultValue":false,"label":"Check box","run":"Section"}
%---
%[text:image:519b]
%   data: {"align":"baseline","height":203,"src":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhMAAAG3CAYAAAAOxtdBAAAA4WlDQ1BzUkdCAAAYlWNgYDzNAARMDgwMuXklRUHuTgoRkVEKDEggMbm4gAE3YGRg+HYNRDIwXNYNLGHlx6MWG+AsAloIpD8AsUg6mM3IAmInQdgSIHZ5SUEJkK0DYicXFIHYQBcz8BSFBDkD2T5AtkI6EjsJiZ2SWpwMZOcA2fEIv+XPZ2Cw+MLAwDwRIZY0jYFhezsDg8QdhJjKQgYG\/lYGhm2XEWKf\/cH+ZRQ7VJJaUQIS8dN3ZChILEoESzODAjQtjYHh03IGBt5IBgbhCwwMXNEQd4ABazEwoEkMJ0IAAHLYNoSjH0ezAAAACXBIWXMAABcRAAAXEQHKJvM\/AAAgAElEQVR4nOzdeUCN6fs\/8Hed9rJEKqLs2bcMBsXYxlaMbZAtW8kuGUsj2ZpJtFjK3iaj0AhZkhZKjS2pkSihaN9L6zm\/P3z1mz5m8RzVfZ66Xn+NszzP+5ypznXu+36uW0okEolACCGEECImadYBCCGEEMJvVEwQQggh5KtQMUEIIYSQr0LFBCGEEEK+ChUThBBCCPkqVEwQQggh5KtQMUEIIYSQr0LFBCGEEEK+ChUThBBCCPkqVEwQQggh5KtQMUEIIYSQr0LFBCGEEEK+ChUThHDk7++PV69esY5BCCESg4oJQjgQiUQ4deoU4uLiWEchhBCJQcUEIRw8efIEQUFBiI+PZx2FEEIkBhUThHBw584dFBUV4fHjxxCJRKzjEEKIRJAS0V9EQr6IUCjE9OnT8fvvv0NGRgYvX76Ejo7O3z62pKQEVlZWiI6OhkAggIyMDMrKymBpaYkJEybUc3JCCKlbMqwDEMIXHz58gIaGBiZPnoyKigoUFRX942OTkpIgLy+PGTNmQElJCR4eHhAKhejZs2c9JiaEkPpBIxOEfCGRSISKigpERkaibdu20NHRgUAg+NvH5ufnQ0lJCbKysrh9+zb8\/PywZcsWtGnTpp5TE0JI3aM1E4R8ISkpKcjJyeH3339HWlraPxYSANCsWTPIysri5s2b8PLygoWFBRUShJAGi0YmCOEoJSUFzZo1Q5MmTf71cdeuXYO7uzu2b9+OHj161FM6QgipfzQyQQhHsbGxePfu3b8+5smTJ7h27Rrs7e3Ro0cPlJeX4\/379\/WUkBBC6hctwCSEozdv3kBdXf0f77937x7mzZsHXV1deHt7o7i4GGlpaVi0aBFat25dj0kJIaR+UDFBCEfGxsb\/ul7i2bNn6NOnD2RlZXHv3j0IhUJ07NgRgwYNqseUhBBSf2jNBCEcWVtbY+LEiRg8eDDrKIQQIhFozQQhHPXr1w8tW7ZkHYMQQiQGjUwQwlF6ejoUFBTQrFkz1lEIIUQi0MgEIRzZ2dkhOjqadQxCCJEYVEwQwtGYMWPQvn171jEIIURiUDFBCEcVFRUQCoWsYxBCiMSgYoIQjqKiov6zaRUhhDQmtACTEI5yc3MhIyPzn+20CSGksaCRCUI48vDwwIsXL1jHIIQQiUHFBCEcSUtLQ0aGmscSQsgnNM1BCEcvXryAhoYGmjZtyjoKIYRIBBqZIIQjFxcXxMTEsI5BCCESg0YmCOHo4cOH0NDQQNu2bVlHIYQQiUAjE4RwpKSkBAUFBdYxCCFEYlAxQQhHrq6u+PPPP1nHIIQQiUHTHIRwlJaWBmVlZeozQQgh\/4dGJgjh6OHDh0hPT2cdgxBCJAYVE4RwlJKSgtzcXNYxCCFEYtA0ByEc5eTkQE5ODioqKqyjEEKIRKCRCUI4cnJyogWYhBDyF1RMEMKRrq4uLb4khJC\/oGkOQjhKTU2FkpISVFVVWUchhBCJQCMThHBkZ2eHp0+fso5BCCESg0YmCOEoIiICHTp0QOvWrVlHIYQQiUAjE4RwlJiYiPz8fNYxCCFEYlAxQQhHz58\/R1ZWFusYhBAiMWiagxCOCgsLIS0tDWVlZdZRCCFEItDIBCEc+fj44NWrV6xjEEKIxKBighCOioqKUFlZyToGIYRIDJrmIISj4uJiSEtLQ1FRkXUUQgiRCDQyQQhHLi4uiImJYR2DEEIkhgzrAITwzYgRI9CqVSvWMQghRGLQyAQhHMnLy9PeHIQQ8hdUTBDCkaurK7XTJoSQv6AFmIRw9PLlSzRv3hxqamqsoxBCiESgkQlCOEpISKB22oQQ8hdUTBDC0Z07d5CRkcE6BiGESAya5iCEoz\/\/\/BOamppo0aIF6yiEECIRaGSCEI68vLzw\/Plz1jEIIURiUDFBCEcDBw6kUQlCCPkLmuYghKOcnBzIyclBRUWFdRRCCJEINDJBCEc7d+7Eo0ePWMcghBCJQSMThHB0\/\/59aGtrQ0NDg3UUQgiRCDQyQQhHysrKrCMQQohEoWKCEI4uX76Mly9fso5BCCESg6Y5COEoNzcXsrKytACTEEL+D41MEMLRxYsX8fbtW9YxCCFEYlAxQQhHWVlZKCoqYh2DEEIkBk1zEMJRQUEBBAIBLcQkhJD\/QyMThHDk7OyM2NhY1jEIIURiUDFBCEft2rVDs2bNWMcghBCJQcUEIRx16tSJiglCCPkLKiYI4cjX1xcJCQmsYxBCiMSgBZiEcPTq1Ss0adIEampqrKMQQohEoJEJQjiKj49HQUEB6xiEECIxqJgghKOQkBCkpaWxjkEIIRKDpjkI4Sg3Nxfy8vJQUlJiHYUQQiQCjUwQwtHVq1dpoy9CCPkLKiYI4UhBQQEyMjKsYxBCiMSgaQ5COMrNzYWCggIUFRVZRyGEEIlAIxOEcLRr1y48fPiQdQxCCJEYNDJBCEdRUVHQ0tJC27ZtWUchhBCJQCMThHDUtGlTyMnJsY5BCCESg4oJQjg6cuQI\/vzzT9YxCCFEYtA0ByEcRUVFoX379tDQ0GAdhRBCJAKNTBDC0ZMnT5CVlcU6BiGESAwqJgjhKD8\/HyUlJaxjEEKIxKBpDkI4ys\/Ph0AggIqKCusohBAiEWhkghCO9u3bh6dPn7KOQQghEoOKCUI46tWrF9TU1FjHIIQQiUHTHIRwlJGRAVlZWaiqqrKOQgghEoFGJgjh6Pjx44iLi2MdgxBCJAaNTBDCUVJSEpo2bUpTHYQQ8n9oZIIQjt6+fYuysjLWMQghRGJQMUEIRwEBAXj16hXrGIQQIjFomoMQjgoLCyEtLQ1lZWXWURqN6OhonDt3Dl27dsWCBQsgEAi++pifupjSdBUhX49GJgjh6NKlS0hOTmYdo9EoKirCunXrEB0dDWVl5VopJABgzZo1CAwMrJVjEdLYUTFBCEcJCQkoKipiHaPeiUQiZGdno6Kiovq20tJSVFVVVd+fmZlZfX9OTg4SEhLw5s2b6sdnZ2fXWG9SVVWFjIyM6n\/n5eUhISEB6enp1bc9ffoU7969g5mZGaZMmYK8vDwUFRUhKSkJ2dnZ1cdJTExEUlISPg22VlVVIScnB+Xl5UhNTUVSUhIqKysBACkpKbh\/\/z4SExNRWFhY228VIY2ODOsAhPCNsbEx1NXVWceod0KhEDt27EDPnj1hZmaGsrIyrFy5EsuWLcOQIUMQERGBnTt3wsPDA\/7+\/vD29oa8vDzS09MxdOhQODg44MiRI\/jw4QP27t0LALh+\/TrOnTsHDw8PXLhwAUeOHIGsrCxKSkowd+5cLFu2DJ6enkhNTcUvv\/wCkUiE06dPo7KyEqWlpTA2NsawYcOwefNmZGZmoqysDN27d8f+\/fuhoKCAefPmQVpaGgoKCnj58iW6deuGY8eOITAwEO\/evYO3tzeGDh2KUaNGMX53CeE3GpkghKOjR48iNjaWdYx6JxAIMGLECFy\/fh0A8OrVK1y5cqW6tfjly5cxePBgyMrKIjAwECtXrsSVK1fg7OyM8+fP4\/Hjx5g8eTICAwORlpYGAAgMDMS3336LN2\/ewMnJCatXr8aVK1fg4OAADw8P3L17Fxs3bkTHjh2xZ88eTJgwAcnJyZCXl8eZM2dgZGQECwsLVFVVwcfHB76+vsjMzMTu3bsBfByBqKqqgqOjI9zd3fHw4UOEhoZi9uzZ6NSpE0xNTaGvr8\/mDSWkAaFighCODA0N0a5dO9YxmJg0aRLk5eXx+PFj+Pv7w8DAAM+ePUN6ejpev36NWbNmoUWLFjh+\/DhatGgBT09P3LhxA5WVlcjMzET\/\/v3RpUsX3Lx5E7m5uUhISMDEiRMRGxuLN2\/e4ObNm1i\/fj08PDyQnp6Op0+fomXLllBQUICamhrk5eUhLy+PkSNHQlNTEx8+fMCbN29gbW2NNm3aoH379li4cCGCg4ORl5cHGRkZzJw5E23btkXfvn3RpUsXpKamQlFREQoKCmjevDlkZWVZv62E8B5NcxDCkZqaGhQVFVnHYEJRURFz586Fn58fYmNjsXr1apw4cQKHDh2Crq4uevXqhdTUVJibm0MkEmHgwIGQkpKCkpIShEIhAGD27Nm4fv06kpOT0bJlS+jo6CA8PByKiooYPHgwpKSkIC0tjbFjx6Jv374oLCyEUCisXovx6X4ANW77REZGBkKhEEKhEAKBoHoNRXl5efVtwMc1FXQxGyG1g0YmCOHo2LFjiI+PZx2DmVGjRiE6OhoCgQDfffcd2rVrB3d3d4wcORIAEB4ejtDQUNja2mL79u3o1asX8vLyqouJ0aNHo6ioCMePH8e0adMAALq6upCSkoKamhoWLFiAsWPHIjQ0FJmZmRAIBBAKhdUf\/J8KBQDQ1NREy5YtceDAARQXFyMvLw9nz57FN998g2bNmqGysrL6sZ+e++k4lZWVKCgoqK+3jZAGjUYmCOFo5cqVaNWqFesYzDRp0gS9evWCtrY2AGDkyJEIDw9Hnz59AADffPMNBg8ejGXLlqFNmzZo2bIlNDQ0qht9KSsrQ1dXFwkJCZgwYQIAQE9PD+bm5ti+fTtcXV2RmZmJXr16oX379vjw4QOAmqMPnygpKcHGxgbbt2\/H5MmTUVlZiebNm8Pa2rp6hOKvpKSkqm8bOnQonJ2d0b59exgZGdXNm0VII0FNqwjhKDQ0FFpaWujcuTPrKMwUFhZCRkYGioqKKC8vR05ODjQ1Navvz8\/PR1xcHGRkZNCrVy\/k5ORAJBJVrzXJz8\/Hhw8fajwHAJKTk\/H27Vs0adIEvXv3hkAgQHl5OZKTk9G2bVsoKSnhxYsXUFVVrdFs6tP5pKWl0adPHygpKaGyshKJiYlQU1NDy5YtIRKJ8OrVKzRr1gwtW7ZEYWEhnj17htatWzfaNTCE1BYqJgjhyMHBAcOHD8c333zDOgohhEgEKiYI4ehT0yV5eXnGSQghRDLQAkxCONq\/f391bwVCCCFUTBDCWadOndCkSRPWMQghRGLQNAchHOXk5EBWVpYKCkII+T80MkEIR7a2tnj8+DHrGIQQIjGomCCEIwMDg+oeC4QQQqiYIIQzFRUVyMhQvzdCCPmEiglCOPL390dSUhLrGIQQIjFoASYhHGVmZkJOTg7NmjVjHYUQQiQCjUwQwtHly5fx9u1b1jEajcLCQqSnp7OOQQj5F1RMEMJRWlpadRdMUvcsLCzg6+vLOgYh5F\/QNAchHOXm5kJGRob6TNQDX19f\/PLLL\/Dz86MraAiRYDQyQQhHp06dQlxcHOsYDV52djZsbW1hampKhQQhEo6KCUI4GjBgADQ0NFjHaPBOnTqF1q1bY\/HixayjEEL+AxUThHCkra2N5s2bs47RoKWkpMDf3x+bN2+mnh6E8AAVE4Rw5ODggJiYGNYxGrTdu3ejU6dO0NfXZx2FEPIFqOQnhCMLCwuoqqqyjtFgBQUF4e7du\/Dz82MdhRDyhWhkghCO0tLSUFBQwDpGg1RRUQFXV1csXboUXbp0YR2HEPKFqJgghKNz584hOTmZdYwG6eLFi8jPz6dFl4TwDPWZIISjlJQUNG\/eHCoqKqyjNCgZGRmYOHEiVq5cCRMTE9ZxCCEc0MgEIRx5eXnh+fPnrGM0OM7OztDW1sb8+fNZRyGEcEQLMAnhqHXr1pCXl2cdo0GJj4\/H7du34erqSpeCEsJDNM1BCEeZmZmQl5dH06ZNWUdpMBYtWoSmTZvC2dmZdRRCiBhomoMQjvbt20d9JmpRQEAAHj58iA0bNrCOQggRExUThHD03XffoV27dqxjNAglJSXYv38\/lixZgvbt27OOQwgRExUThHDUsWNHWjNRSzw8PKCoqIhVq1axjkII+QpUTBDCkaenJ16+fMk6Bu\/l5OTg7NmzWLVqFS26JITnaAEmIRxlZ2dDVlaWFmB+pQ0bNiA1NRVnz56FtDT\/vtfk5ubiyZMnGD58OBVDpNHj328wIYz5+fnh7du3rGPw2v379+Hv748tW7bwspAAgO3bt8PNzY11DEIkAj9\/iwlhKDs7G6Wlpaxj8JZIJIKzszMWLVqEfv36sY4jloiICERGRuKnn3764lGJnJwcxMXFIScn57P78vPzce\/ePZSVlX3RscrLyznlJaSu0dgcIRytWrWKt9+mJcG1a9eQkpICBwcH1lHEIhKJcOTIEUyfPh3du3f\/oudUVFTg2LFjOHPmDE6dOoUWLVrUuD8jIwPnzp2Drq7uvy7uLSgowMmTJ\/HmzRuoqKhg4cKF6Ny581e9HkJqA\/1FJIQjFxcXPH36lHUMXsrLy8PevXsxc+ZMqKmpsY4jFh8fHyQlJXHajExWVhaTJ0+GlpYWtLW1P7u\/S5cucHR0\/KzI+F+nTp1CSkoK9uzZgx49esDMzAxZWVmcXwMhtY2KCUI4UlVVRZMmTVjH4CVXV1coKytj0aJFrKOIJSMjA3Z2dli8eDHU1dU5Pff27dvo1KkTNDQ0EBoaCn9\/fyQnJ0MkEiEpKQl\/\/PEHKisrkZWVhcjISKSkpODs2bOIi4urPkZcXBwGDRoEJSUljBgxAhkZGcjOzq7tl0kIZ1RMEMLRqFGjoKWlxToG76SkpODq1auwtraGkpIS6zhiOX36NDp27PjFoxIJCQlwdnbGoUOHcO7cOQwfPhyBgYHw9\/dHRUUFjh49ipSUFOzevRsrV65EUVERjh49iilTpuDu3btISUnBggULkJiYCODjFEtFRQWSk5Nx8uRJGBgYQFdXty5fMiFfhIoJQjg6cOAAHj9+zDoG7+zevRt9+vTB0KFDWUcRS3JyMoKCgrBjx44vWjMTExMDe3t76Ovro1evXnj37h0MDAxw9+5dJCcnY9y4cdi4cSO0tLQwdepUiEQiyMvLY8yYMWjdujWGDBkCS0tL5ObmVv+8SUtLQyQS4cOHD1BXV0dJSQmSk5Pr+JUT8t+omCCEo\/Xr16Nnz56sY\/BKWFgYrl27huXLl7OOIrYtW7bg1atXePr0Ka5du4br16\/Dz88P7u7uyM\/Pr\/HY8vJy7Nu3D\/r6+ujfvz\/ev3+PESNGQEtLC0uWLEFGRgZ69+6NwMBASEtLQ0VFpbpAUVZWRpMmTSAjI4PKykrIy8tXX70hEokgEAjQvXt3mJqaorCwEC4uLvX+XhDyv+hqDkI4SklJoX0kOCgrK4OdnR3MzMzQt29f1nHEEhQUhNjYWMjIyGDOnDnVtzdr1gy7d++GoqJijcdHRUUhOTkZP\/zwA0pKSuDh4QEjIyMAgEAgQGBgII4dO4atW7eiY8eO1cWEQCCoPoaUlBQEAgGkpKRqXH4qJydX\/d8KCgooKCioq5dNyBejkQlCOPr999\/x+vVr1jF448KFC6iqqsLq1atZRxFLeXk5Dh48CAsLC5w8eRIqKirV9y1fvhyrVq2q8QEPAKWlpaisrMTr168REhKCwsJCyMrKAgBOnjwJX19fGBsbo0ePHigtLcXr16+Rnp6O7OxsZGVl4e3btygoKMD79++RkZGBjIwMAB83Rrt9+zaKiooQGRmJtLQ0zJ8\/v\/7eDEL+gWDHjh07WIcghE90dHTQqVMnKCgosI4i8fLy8rBx40YsX74c\/fv3Zx1HLJ6ennjw4AFsbW3RoUMHpKam4v79+xg5ciSKi4sREhKCdu3aQVNTs\/o5WlpaqKqqQmZmJoYMGQINDQ0MGDAAGhoaEAgEyM3NxYMHD2BoaIgBAwYgLCwMqqqqUFJSQm5uLmRkZCAnJ4fU1FQoKipCTU0NvXr1QlhYGPLy8lBaWornz59j3rx5GDZsGMN3h5CPaG8OQjiys7PDuHHjeNu9sT5t27YNMTExuHDhwmff3vkgLS0NY8eOxebNm2FsbAwAePPmDfT19WFtbY3x48fj4MGD8Pf3h6GhITZt2vSfvSI+EQqFny3k\/OttQqEQAKr\/LRKJsGjRIhgZGWHKlCm0HwiRKDTNQQhHGhoa1AHzC8TGxuLSpUvYsmULLwsJAHB0dETnzp0xd+7c6tu0tbVhYWGBtm3bok2bNrC1tYW7uzsSEhIwadIkeHh4fFFb7L\/7GfrrbdLS0jX+LRQK0aRJE6ipqVEhQSQOjUwQwlF6ejoUFRVp19D\/sHz5crRo0QK\/\/PIL6yhiiY6OxrJly3D69Gn06tWrxn1lZWWQkpKqUSQJhUJcuHAB7u7uqKqqgrm5OQwNDWstj0gkQnp6Opo3b05TbETi0NcrQjjas2cPoqOjWceQaGFhYfjzzz+xatUq1lHEdvDgQYwePfqzQgIA5OXlPxttkZaWxsyZM3H+\/HmMGjUKlpaWWLFiBeLj42slj5SUFDQ1NamQIBKJiglCOJo6dSo6dOjAOobEKikpwa5duzBlyhS0bduWdRyxXL58GTExMWIVQwoKCrC0tERAQADk5OSwbNky7Nixo\/qKDEIaIiomCOGoffv29O3wX5w+fRo5OTlYsWIF6yhiKSwsxL59+7Bw4cKvKoY6duwIJycnODg44N69exg7dizOnj0LmlkmDREVE4Rw5OHhgT\/\/\/JN1DImUnp4OX19f7N69u0Y\/Bj7x9PSEiooKli1bVivHGzhwIK5du4a1a9fi9OnTmDNnDkJCQmrl2IRIClqASQhHGRkZkJOTQ\/PmzVlHkTjbt29HSkoKTp06xTqKWNLT0zFjxgzY2Nhg1KhRtX78vLw8HDlyBGfPnsXw4cOxefNm6Ojo1Pp5CKlvNDJBCEe3bt1CZmYm6xgSJyIiAp6enjA3N2cdRWy2trZo3749Ro4cWSfHb968ObZu3YqLFy+ipKQEP\/74Iw4cOIC8vLyvOm5JSQmSkpJqKSUh3FExQQhHb9++RW5uLusYEqWqqgoHDhzA9OnTMXDgQNZxxBIeHo6AgABYWFjUeR+RLl26wN3dHba2toiMjMSECRPg5eUl9vFsbW3h7OxciwkJ4YamOQjhqLi4GFJSUlBSUmIdRWIEBATA3t4eFy5cgKqqKus4nFVVVcHY2Bj9+vXD5s2b6\/XcQqEQ7u7ucHFxgY6ODjZu3IjBgwd\/8fMfPnyIadOmwdPTEwYGBnWYlJB\/RiMTDUhKSgrCw8PrdbX4o0ePEBUV9Y\/3C4VCXLt2DY8ePfrsvpKSEgQFBSE9Pb0uI\/6n1NRUREREoLKy8ose7+rqimfPntVxKv4oLCzEgQMHMH\/+fF4WEgBw9epVZGVlMdkiXVpaGiYmJrh8+TJ69OgBCwsLrFmzBomJiTUe5+3tjdTU1M+e7+LigpkzZ1IhQZiiYqIBCQ8Ph62tbb0WE8eOHYOTk9M\/3i8UCnHr1q2\/bfKUnZ0NS0tL5g2g\/vjjD\/zyyy9f1AIZ+LgFNLUz\/v8OHz6MqqqqGltz80lOTg4sLCwwZ86cL95Xoy5oaGjAxsYGHh4eKCgowJgxY+Dg4IAPHz4AAJydneHn51fjOcHBwYiPj4elpSWLyIRUo2KiARGJRJCVlYW0tDSKiopq3PfpW3dVVRVKSkpq3P6\/j\/2ktLQUhYWFf3vfp+fIycnVKF4qKytrPEdGRgb29vYwMTGpvk0oFKKkpASysrKQkZGp8XyRSPSP5\/xfhYWFNQoAkUiEiooKAB+3jf70R\/ivKisrq19\/RUVF9WZKn57\/19f+1\/fprwwMDKClpfVFGRu6xMREnDt3DlZWVrztvXHo0CFoa2tj9uzZrKMA+Nifws3NDfb29ggJCcH8+fPh6uqKly9fws3NrXqxZkFBAXbu3IkZM2ZAQ0ODcWrS2FEx0YDIycmhqKgIlpaW6NevH3744QfExsYC+NjRb82aNRg5ciTMzMxQVVUFf39\/GBgYoH\/\/\/jA2Nq5eDV5YWAhbW1sMGTIE\/fv3x5w5c5CQkADgYxGxbt069OvXD8bGxnj+\/Hn12oFr165h9OjR0NPTw6RJk\/DkyRMAwIoVK+Dq6grg4yjAmDFjMHjwYOzatQtlZWWQlZUFAISGhmLcuHHo378\/Zs6c+Y+r08PDwzF58mTo6elh8ODBOHHiBEQiEQoKCrBmzRps2bIFo0ePxoABA7Bv377qgiMoKAgjR47EoEGDsGPHDsybNw9Pnz6FoqIipKWlISsri\/LyctjZ2WHIkCEYNGgQbGxsPisqjh8\/Xv2+NnaHDx+Gvr4+Ro8ezTqKWBITE3H9+nXs27cPysrKrOPUMH36dJw\/fx5z5syBi4sLcnJy8PDhw+qFmidOnAAALF26lGVMQgBQMdGgyMrKIiIiAoWFhXB1dYWioiJWr16NkpISpKWlwc3NDT\/88AM2bNiAkJAQrF69GtOmTYOrqyvy8\/OxaNEiFBcX4\/r16\/Dz88PevXvh4eGB58+fw9raGiKRCHv37kV4eDgOHjwIPT093LlzB8rKykhNTcWyZcswdOhQnDt3Dnp6eoiOjkZVVRWePXuGjIwMZGZmYtGiRejVqxccHByQmZmJ58+fQ0VFBc+ePcPatWthZGQEX19fdO7cGWvWrEF2dnaN11hYWIi9e\/eiZ8+e8PDwwKxZs7BlyxbExMRATk4OISEhCAwMxNatW7FgwQLY2tri8ePHyMzMhKWlJfT19eHg4IDHjx\/Dz88PJSUl1bszysjI4NSpU7hx4wYcHBxw9OhR3Lt3D46OjjVGLVauXIlu3brV9\/9eifPw4UPcuXMHK1euZB1FbI6Ojhg8eDAGDBjAOsrfkpWVhUgkwtu3b6t\/Brdv347z58\/jxo0bsLKy4m1zMNKw0MRvA1JaWor+\/fvDwcEBioqKGDhwIMaNG4e4uDhUVVVhxIgR2LBhAwDA1NQUBgYG2LhxIwCgd+\/eGDJkCIKDgzF69GgMGjQIAJCUlIQWLVogJycHqampCAwMhKWlJSZMmIAJEyYgLCwMJSUlUFBQQLNmzRAZGYkOHTrAxMQEHTp0gFAohJycHBQVFREWFgZ5eXns27cP8vLy0NLSwuPHj1FZWYnAwECkpaUhPz8fgYGBKC0txdWrVxEeHg4jI6Pq16ioqAgnJyeoqakhKSkJTZs2hZSUFN69e4euXbtCWYE7gTsAACAASURBVFkZNjY2mDBhAsaPH4\/z588jJSUFb9++RZMmTWBrawsA0NHRwePHj6unOQQCAQoLC3H58mUIBALcv3+\/+py+vr5Ys2ZN9R\/tlJSURn8lR3l5ObZu3QojIyPo6uqyjiOWW7duISQkBBcvXmQd5W+VlZXB29sbvr6+GDZsGGRlZVFSUoLMzEwsXboUCxYs4O2IEGl4qJhoQIRCIdq2bQtFRUUAgIqKCpo2bYrCwkJISUnVWGmfm5uLrl27Vv+7VatWaNq0KYqLi5GVlQVLS0tkZ2dDU1MTr1+\/Ro8ePZCfnw+hUAhNTc3q52lrayM7OxstW7bEjRs3YGNjA2dnZ9jZ2WH27NnYsmVL9bqInJwcaGhoQF5eHgCgqqoKNTU1lJeXIysrC9LS0nj27BnKysogLy8PMzOzGucCPq75uHDhAi5cuIBWrVpBKBRCIBBAIBBAJBJBTk6u+kO\/vLy8eqFkTk5OjY6Vbdu2RfPmzauLCSkpKZSVlSE3NxfS0tJ49OgRqqqq0KpVK3z33Xc1Mly\/fh2KioqNunOhj48PcnJyeLsraHl5Ofbt24e5c+eiS5curOP8LRkZGYwdOxbTp09HkyZNAHz8HQ8ODsbixYtrrd03IbWBiokGREZGBo8ePUJSUhI6duyIhw8fIj09HTo6OtWjE5\/o6uri9u3bqKiogKysLMLCwpCVlQUtLS38\/PPPUFJSgqenJ+Tk5PDjjz+iuLgYWlpaUFJSQkhICAwMDFBeXo4\/\/vgDvXv3xuvXr+Hv7w8nJycoKSnh9OnTMDMzw7Rp06q3au7WrRuePn2KhIQEdO3aFbGxsXj16hUUFRXRtm1bNGvWDCdPnoSCggJKS0tx6tQpaGtr13iNwcHBOHnyJE6cOAEDAwNcuXIFixYtqr5fKBRWv06RSISqqiqIRCLo6urCxcUFmZmZaNWqFYKDg5GSkgKBQFD9PGVlZWhqaqJ3796wsbEB8PHS18TExBojEZaWltUFW2OUl5eHU6dOYevWrby9FPTMmTMQCoVYu3Yt6yj\/SCAQfLbRmEAgwPHjx7FkyRL07t2bUTJCPkfFRAMiEAiQmZmJtWvXYvDgwfDy8sL48ePRqVMnlJSUVF\/pAADLli1DQEAA5syZg\/79+8PT0xPTpk2Dnp4e1NXVcffuXRw\/fhxv375FbGwsmjZtCmVlZWzatAkbNmxAaWkpsrOzERcXh969e6NJkyb47bffcPXqVYwePRpRUVEYN24cdHR0UFRUhOLiYgwfPhwGBgaYP38+DA0NERQUhPz8fFRVVWHGjBnw8vLChAkTMHbsWAQGBkJVVRVz586t8RpbtmwJaWlp+Pv7IyIiAnfu3IFIJEJ2djakpKRQXl5eo2gqKytDYWEhpk+fjm7dumHu3LkYMWIEgoODAXzcLjo\/Px8lJSVQUlKCqakpVqxYgYyMDLRq1Qq\/\/fYbVq1aVaMj4p07d9C9e3c0a9asjv+PSqYTJ06gZcuWMDQ0ZB1FLLm5ufDw8MCaNWt4N1117tw5vHnzBidPnmQdhZAaBDt27NjBOgSpHSKRCEOGDMG3336LR48eYdasWbCwsIBAIIC0tDQ6dOiA7t27AwCaNWuG8ePHIz09He\/fv8fChQuxfv16yMvLY9iwYRCJRHjx4gWGDBmCNWvWQE1NDT169ECfPn2qRxU+XenRvXt39OvXDxMmTEBBQQFevXqFPn36YM+ePVBVVYVAIEDfvn3RoUMHjB8\/HuXl5Xj\/\/j2MjY3x3XffoVevXmjTpg0MDQ1RVlaG5ORkDB8+HDY2Np99YGtpaaF37974888\/IRQKYWFhgaFDh6J169bQ1taGoqIiBgwYUP2NWUFBAQMGDICKigq6du0KWVlZ5OXlYebMmYiIiMD06dOhra0NNTU1dO\/eHV27dsWQIUOQlJSEwsJCrFy5EvPmzYOUlFR1hvv370NDQ+OzKZjGICYmBtbW1tixYwc6duzIOo5Ydu\/ejeLiYlhZWdV52+zaVFZWBgsLCyxatIhTh0xC6gO10yaNQnh4OMzMzODo6IhBgwZh27ZtCA8Px\/Xr19GqVStOxyopKYFAIKhe+9GYzJgxA8rKynB3d2cdRSz37t3D\/PnzcebMGd59IO\/ZswehoaHw9\/fnbU8P0nDRNAdpFAYMGIAff\/wR1tbWkJeXh7KyMg4ePMi5kAAAa2trTJ06FcOGDauDpJIrODgY6enp8Pb2Zh1FbEePHsWUKVN4V0jEx8fD29sbTk5OVEgQiUQjE6RRycrKQmlpKdTV1asXhnIVFhaG9u3bf7Y4tCGrrKzEtGnT8P333\/O2r8Tt27dhY2MDHx8f3nWMNDc3h6KiIvbv3886CiF\/i0YmSKOipqb21cfQ1tZudI2CDh8+jMzMTMybN491FLHk5+fDxsaGl62no6KiEB0d\/VVblBNS1\/iz+ogQCWFvb1\/dKrwxSElJgZeXF6ysrHh7BcuxY8cgEAhq7BHDBxUVFbC2tsakSZN4u+CVNA40MkEIR4sWLWpUUxzHjx9Hv379MGnSJNZRxPL+\/Xv4+\/tj9+7dvBtROnXqFHJycmBubs46CiH\/ikYmCOEoISGheufGhu7JkycICAjAihUrWEcR2+HDh9GtWzeMGDGCdRROsrOz4ePjg59\/\/pm3zcFI40HFBCEcJSQkICsri3WMOicSiWBjY4Phw4dL7EZY\/+Xu3bs4e\/Ys1qxZwzoKZ8ePH4e6ujpvm4PR2v7GhaY5COFo3bp11W24G7KLFy\/i3bt3cHFxYR1FLGVlZdi3bx\/mzJnDu9bTsbGx8PLy4u17f+HCBaSmpvKyiCPioZEJQjg6evQoEhISWMeoUyUlJXBwcIC5uTnvrn745OrVqygqKoKlpSXrKJyIRCLs2bMH48aNg76+Pus4nOXm5uLQoUO8\/bkh4qGRCUI4atGiRYNvHOTu7g4VFRXMnDmTdRSxFBcXw8XFBYsWLeLdFSg3b95ESkoKHB0dWUcRy7Fjx6Curs7bnx0iHiomCOHohx9+gKysLOsYdSYxMRGHDx\/GgQMHeLs7qqOjI2RkZDBjxgzWUTj5NDUzc+ZMXn6zf\/LkCdzc3HDy5Ele7XtCvh4VE6RRKSsrQ2RkJIqLiyElJQVdXV3O1+8fPnwYY8aMabDttPft24d+\/fph3LhxrKOIJT4+Hr\/99hsOHTrEu2LIxcUFRUVFWLhwIesonAmFQuzbtw9jx47F0KFDWcch9YxKR9JopKenw8zMDIGBgZCTk0NhYSE2btyICxcucDrOrFmz0Llz5zpKyVZUVBRiY2OxZ88e1lHEZm9vD319fd5dCvrmzRteNwcLDQ1FamoqtmzZwjoKYYBGJkij4eHhgZiYGBw8eLC6eVFiYiI2b94MAwODL970Kzc3F82bN6\/LqEwIhUI4ODhg8uTJ0NHRYR1HLH\/88QdiY2N5uRmZs7Mz+vXrh8mTJ7OOwllFRQXs7OwwdepUtG7dmnUcwgCNTJBGIy4uDt26davRBVFHRwd5eXkoLi7+4uOcP38eL168qIuITHl5eeHFixe8azn9yafW00ZGRrxrPR0bG4uIiAhs2LCBdRSxnDhxAoWFhbz92SFfj0YmSKNRVVUFeXl5iEQi5OfnQygUIjg4GFOmTIGWltYXH2fbtm2Ql5evw6T1Lzs7G66urrC0tOTlwj\/g41UEubm5vGs9LRQKsW3bNowYMQI9evRgHYez1NRUuLi4wNbWFk2bNmUdhzBCxQRpNKSkpCAQCCAUCrF582akpqYiOzsbu3bt4nR1RmhoKHr27AldXd06TFu\/Tp48iY4dO2L27Nmso4glMzMTvr6+sLKy4t0U1Pnz5\/H69WscPXqUdRSxHDlyBHp6erzdu4XUDprmILUqPz8fGRkZrGP8o8rKSggEAmzfvh3Ozs4wMjLCunXr8PTp0y8+RkxMDPLz8+swZf1KSkqCr68v777R\/9WJEyfQrl07TJw4kXUUTgoLC3Hy5En89NNP0NTUZB2Hs6dPn+LWrVtYv3496yiEMSomSK366aef0KdPH3h7e+PDhw+s43zm034Bbdq0QYcOHbBo0SK8e\/cOJ0+e\/OJjmJqaomfPnnUVsd5t374dvXr14u3lfE+ePIGnpyfMzc1519vAw8MDioqKvGzwVFlZiR07dmDYsGHo06cP6ziEMX795hGJp6+vDyUlJRgbG8PIyAiXL19mHelfqauro2\/fvnj27NkXb0y0f\/9+REdH13Gy+hEYGIjnz59j586drKOIRSgUwtbWFt9\/\/z2+\/fZb1nE4efnyJY4cOQJzc3PIyPBvxvncuXNITEzkXbtyUjeomCC1ytjYGJGRkdi4cSNCQkJgZGSEpUuXSszVD3\/3zbWyshIqKiqQkpL6omOMHz+el0PS\/6uyshIHDx7E0qVL0a5dO9ZxxBIUFIS0tDRYWVmxjsLZL7\/8goEDB\/KyOVhJSQnc3NywatUquhSUAKBigtQBdXV17Nu3D4GBgZg5cyZOnjyJYcOGwcbGBikpKcxySUtLo7CwsMZtsbGxeP36Nadh5vbt2\/Nukd\/fcXNzQ3Z2NubMmcM6iljKyspw8OBB\/Pjjj2jZsiXrOJxERkYiISEBu3fvZh1FLJ6enlBRUcH8+fNZRyESgooJUmdGjhyJc+fO4cqVK2jXrh127NgBAwMDnD179ounFGqTkpISIiIicObMGURGRiIwMBB2dnZYuXIlp2Li4MGDiIuLq8Okde\/du3dwdHTE+vXreXs536FDh5CdnQ1jY2PWUTipqqrCr7\/+CkNDQ16OCL148QL79++Hqalpg7tEmohPSsTirzppdPLy8uDq6goXFxe8efMGhoaG2LRpE4YPH15vGRYuXIji4mLMmzcPWVlZkJWVRd++fdGvXz9Ox3n06BHatGnD66mOn376CW\/fvoWXlxfvFi0CH1tP\/\/DDD7CxseFdx8iTJ09i\/\/79CAsLg5qaGus4nJmamqK8vBynT59mHYVIEP6t+iG81Lx5c2zevBmzZ8\/GsWPHYGtri4CAACxfvhwbNmyol70uysvL0aRJE0ydOvWrjlNcXCyRV6p8qadPnyI0NBTu7u68LCSAj4tgv\/nmG94VEunp6Th27Bh27tzJy0Li3r17ePbsGc6cOcM6CpEw\/PxLQnirffv22Lt3L0JCQjBhwgS4uLjgm2++gaOjIwoKCursvCKRCGlpabWyaj4sLAxpaWm1kIqNAwcOYNSoUbxtuvXkyRMEBwfzsi\/GiRMn0LlzZ95tjQ58nJ5xdHTE+PHjeTk9Q+oWFROEiREjRuDixYs4c+YMtLW1sX79eowaNQo+Pj51cj6hUIjx48fD0NBQ7GPExcUhMzMTGzduRP\/+\/WsxXf05f\/48oqOjeflBDPz\/3gaGhoa8623w8uVL+Pv7Y926dayjiMXNzQ2vXr3CsmXLWEchkkhECGP5+fmiPXv2iJo2bSoCIBo3bpwoJiaGdaxqxcXFooyMDNG2bdtE3bp1E40fP15079491rE4y8\/PFw0bNkx08OBB1lHEdvbsWdHAgQNF6enprKNwIhQKRdOnTxetWLGCdRSxZGRkiAYNGiQ6e\/Ys6yhEQtHIBGGuadOm2Lp1K6KiorB48WLcvHkT3377LTZu3IjExEQmmdLS0hAREYHff\/8dgYGBiI6ORmFhIRITEyEnJwcFBQUmub6Gl5cXNDU1YWpqyjqKWAoKCnDixAmsW7cO6urqrONwcu3aNSQmJmLTpk2so4jFzc0NXbt2xaxZs1hHIRKKruYgEufu3bvYvn07goOD0a5dO1hZWWHhwoX1chlaUlISHjx4gIKCAujo6KBLly5QVVWFsrIybty4AeBjl0++FRTv3r3DjBkzsGvXLowePZp1HLEcPnwYQUFB8PHx4VXHyPLychgaGmLmzJlYunQp6zicvXz5EjNnzsTBgwfr9eorwi9UTBCJVF5ejhMnTuDQoUN49uwZRo4cifXr18PIyKhOzpeUlITHjx+jrKwMnTt3Rp8+ff6xWLCyssL48eN59Yd1xYoVyM7Oxrlz576406ckSUxMxJQpU+Dk5MS7YsjNzQ0+Pj44f\/48lJSUWMfhzMTEBHJycrzd1ZTUDyomiERLS0vDsWPHYG1tDQCYNm0arK2ta23xXXl5OQICAlBQUICBAweia9eu\/\/mt9+bNm+jevTtvVrSHh4djzZo1OHPmDLp168Y6jlgWL14MkUjEu94GqampmDhxInbt2lVnhXBdCggIwNatW3H16lVoaWmxjkMkGK2ZIBJNU1MT27dvx4MHDzBnzhxcvHgRQ4cOxdatW\/H+\/fuvOvb79+\/h4+MDKSkpzJo1Cz169Pii4fOmTZtCVlb2q85dX0QiEVxcXDB37lzeFhLh4eG8bT1tb2+Pzp0787KQqKqqgoODAxYtWkSFBPlPVEwQXtDT04O3tzdu3rwJPT092NraQl9fH6dPn0ZZWRnn48XFxeH3339Hv379MGXKFE7rH3x8fPD8+XPO52TB3d0dL1++xKJFi1hHEYtQKISzszOmTp3Kuw+06OhoREVF8XZHVg8PD0hLS2PJkiWsoxAeoGKC8MrYsWNx69YtODk5obKyEosXL8aoUaMQERHxxcdISEhAcHAwJk6ciF69enHOsHXrVs4tuFnIzs6Gk5MT5s2bx7uNsD5xc3PDy5cvsXjxYtZROBGJRHB0dMSYMWPQs2dP1nE4S0tLg729PZYtW4YmTZqwjkN4gIoJwjuysrJYs2YNwsLCsGnTJjx+\/BjDhg3DsmXLEBMT86\/PffHiBYKCgjBlyhTo6OiIdf7g4GCkp6eL9dz65OLigs6dO2PlypWso4jl\/fv3cHV1haWlJVq0aME6Dife3t54+vQpzMzMWEcRi729Pfr06YPp06ezjkJ4gooJwlva2tr49ddfERUVhalTp+LEiRMYPnw47O3tP2vNHRkZiaCgINy7dw9jxoz5qsWT0dHRyM7O\/tr4derVq1e4evUqtmzZwsurNwDg9OnT6NmzJ2bPns06Cie5ubk4fPgwVq1ahTZt2rCOw1lsbCwiIiKwY8cO3v7skPpHxQThvd69e8PPzw9ubm7Q1dWFpaUlRo0aBU9PT5SUlAAA7t+\/j1mzZuH58+dQUVH5qvNZWFiINT1Sn+zs7KCnp4cBAwawjiKW+Ph4+Pj48LLtt5eXF9q2bYv58+ezjiIWJycnXu\/dQtigYoI0GAsXLkRQUBCcnJyQnJyMBQsWYOrUqXj8+DEKCwshFAqhqKj41d+23N3dER8fX0upa9\/169cRHByMDRs2sI4itp9\/\/hmDBw\/GN998wzoKJ8nJyfDy8oK5uTmvGmt9cv78edy9e5e3XVIJO9RngjRI8fHxsLe3h7e3N6SkpNC5c2dYWlpi3rx5X33sa9euoXPnzujSpUstJK1dJSUlmDZtGsaOHQsLCwvWccQSEBCAn3\/+GZcuXULbtm1Zx+HExMQElZWV8PT0ZB2Fs7y8PEyZMgXGxsZYvnw56ziEZ2hkgjRI3bp1w4kTJ3Dr1i18\/\/33iImJwc6dO2FnZ\/fVW5137NhRYhcEXrhwAVJSUli9ejXrKGIpKyuDk5MTzM3NeVdIBAcHIyYmBtu3b2cdRSxeXl5QV1enS0GJWKiYIA3a0KFD4evri1OnTkFeXh4\/\/fQT9PX1cenSJbGP6eTk9J9XjbCQnZ2NY8eOwczMDHJycqzjiMXLywtSUlIwNjZmHYUToVCII0eOwNjYWCJHrP7L69ev4erqiuXLl0MgELCOQ3iIignS4AkEApiYmODWrVuwt7fHu3fvMHXqVEyfPh1\/\/PEH5+OZm5tL5ALMHTt2oFWrVpg4cSLrKGJ5+\/YtDh48CHNzc15togYAfn5+yMrKgomJCesoYtmzZw\/69u3Lu31PiOSgYoI0GhoaGrCwsMD9+\/excOFCXLx4EcOGDcPKlSuRmZn5xcepqKgQq+tmXYqOjsaVK1ewdetW3rT6\/l\/79+9Hz549edd6Ojs7Gzt37sTSpUuhqqrKOg5n4eHhiIyMhI2NDaSl6SOBiId+ckij0759e7i5ueHGjRsYNmwYjhw5gsGDB8PFxQWFhYX\/+Xx3d3ckJSXVQ9Ivd+jQISxcuBADBw5kHUUsjx8\/RmRkJLZu3co6Cmf79u2Djo4OZs2axToKZ0KhEI6OjjAxMUHnzp1ZxyE8RsUEabTGjRuHmzdv4syZMxCJRDA3N8fEiRMRFBT0r8\/76aefJOpD+9q1a4iLi+P15XzOzs4YP34871pPv3jxAgEBAbC2tubliJCvry\/S09OxcOFC1lEIz1ExQRo1OTk5zJ07F5GRkVi\/fj1iYmIwZswYLFmyBNHR0X\/7HG9vbyQkJNRz0r+Xm5uLHTt2wNjYGK1bt2YdRyyenp54+PAhL4shR0dHjBs3Dnp6eqyjcJaVlYUDBw7AxMREYq9OIvxBxQQh+Lie4sCBA7hz5w4WLlyIU6dO4bvvvsOWLVs+W0\/RokULiZlbdnNzg7q6OpYtW8Y6iljy8vLg5OQEMzMz3hVDV69eRWhoKG8vwz148CDatGnDuytniGSSjL+IhEiIPn36wM3NDZcuXULXrl3xyy+\/QF9fH25ubigtLQUATJ8+HR07dmSc9OPOjr\/\/\/js2btwIeXl51nHE4uHhgU6dOvGubXZRURHs7OywdOlSsTeMYykxMRHXr1\/Hli1beHsZMZEsVEwQ8jeMjIwQGBgIV1dXlJSUwMTEBEZGRrh9+zYOHz6Mly9fso6I3bt3o0uXLjAwMGAdRSxJSUnw9vbmXSEBAOfOnYOqqiovswMfr5wZNmwYBg0axDoKaSD41zyekHrStGlTmJqawtDQELt374abmxtGjx6N1q1bY\/DgwUyz3b17F4GBgdUdL\/lo69at6Nq1K0aMGME6CidZWVlwc3PDpk2bePmt\/ubNm7h58yZu3LjBOgppQGhkgpD\/0KZNGxw5cgS3b9\/GtGnTkJWVhaVLl2Lv3r3Iysqq9zyVlZVwcnLCggULJLJ51pe4ffs2Xrx4gZ9\/\/pl1FM527doFTU1NTJo0iXUUzkpKSuDg4ABTU1N06tSJdRzSgFAxQcgXGjJkCM6fP485c+YgNzcX27Ztw\/Dhw+Hn51evOfz9\/ZGVlYU1a9bU63lrS1VVFVxdXWFiYsK71tNRUVEIDg7G9u3bJWYRLhfnz5+HQCDg7c8OkVz8+20ghCEpKSk4ODggKioKBw4cQE5ODqZNm4bp06cjLCyszs9fUFCAgwcPYuHChWjSpEmdn68u+Pr6IjMzs1Z2cK1vR48exYwZM9C7d2\/WUTh79+4djh8\/jmXLlvF2wS6RXFRMEMLR1atXISsri\/Xr1yMqKgpr167FxYsXMWLECKxatQpv376ts3Pb29tDXl6et5fzZWRkwM7ODsuXL0fz5s1Zx+Hk8uXLePbsGW8vw929ezcUFRUxefJk1lFIA0TFBCEcJSUlISMjAwDQoUMHODo64tatWxg5ciQOHz4MfX19HD58uNbXU8TGxuL8+fOwsrLiZbdFADh8+DA6dOiAmTNnso7CSV5eHnbs2IH58+fzrh8GADx69Aj37t3Dr7\/+SruCkjohJRKJRKxDEMIn+fn5EAgEUFFRqXF7eXk5Ll68iD179iA2NhYDBw7Epk2bau2Dc\/Xq1ZCWloaTk1OtHK++vXjxAnPnzsWxY8fQv39\/1nE4+fXXXxEeHg4fHx\/e7WgKAEuWLEH79u15ueCV8AONTBDCkbu7O+Lj4z+7XU5ODrNnz0ZQUBCsrKyQmJiIWbNmYdasWf\/YmvtL3blzB1FRUVi3bt1XHYclBwcH6Ovr866QSE5OxtmzZ7FlyxZeFhKXLl3CkydPsHTpUtZRSANGxQQhHMnJyf3rh4q6ujp27dqFkJAQmJmZwdfXFyNHjoSlpSVSUlI4n6+oqAjW1taYOnUqOnTo8DXRmbl69SrCwsJ4WQwdOnQIenp6+Pbbb1lH4aygoAD79u3D4sWLeTk9Q\/iDiglCOJo4ceIXtVDu06cPXFxcEBYWhm7dusHe3h4DBw6Ek5NTdWvuL\/Hbb78BANavXy92ZpYKCwuxc+dOLFu2DNra2qzjcBIWFgZ\/f39s2rSJdRSxHD9+HEpKSrxdNEr4g4oJQjiyt7fHo0ePvvjx+vr6CAkJgYuLC5SUlLBu3TpMnjwZ165d+8\/n5uTk4Ny5c9i4cSMUFRW\/JjYzvr6+aN68OZYvX846CicfPnyAnZ0d5s+fD11dXdZxOHv\/\/j38\/Px4vWCX8AcVE4RwtGLFCnTr1o3TcxQUFGBmZoaIiAjs2rULkZGRmDhxImbMmPGv25k7OjpCVVUV33\/\/\/dfGZiIrKwvu7u5Yu3Yt74qhS5cuoby8nJdTMwBgZ2eH7t2783bvFsIvVEwQwtGHDx8g7kVQmpqasLKyQmhoKKZOnYoLFy5g0KBB2Lt3L9LS0mo89sGDB\/Dx8cG6det4eznfzp07oa6ujvHjx7OOwklubi5OnjwJExMTXjYHi4yMxLVr16jTJak3VEwQwtGpU6fw\/PnzrzqGnp4e\/Pz8cP36dfTs2RPbtm3DqFGjcOrUqerHODo6YtasWRg6dOjXRmbi3r17CAsLg7W1Ne9aT+\/fvx8yMjKYNm0a6yicVVRUYM+ePZg3bx4vO3USfqI+E4RwlJaWBhUVlc\/6TIirtLQULi4u+PXXX5Geno5p06bh22+\/RXh4ODw9PWvtPPXNxMQEXbt2xZYtW1hH4SQ2NhZz586Fq6srLws5Pz8\/ODk54dKlS2jWrBnrOKSR4NfXBUIkwN27d2u1ZbaCggLWr1+Pe\/fuwcLCAjdu3IClpSVyc3M\/m\/rgC39\/f7x8+RImJiaso3Dm4uICQ0NDXhYS+fn5OHLkCExNTamQIPWKiglCOMrKysKHDx9q\/bgdOnSAvb091q5dCxUVFURGRqJfv37Yu3cvcnJyav18dSU\/Px8\/\/\/wz5s2bB01NTdZxOAkNDcWDBw94d+XJhQ8XhgAAIABJREFUJ3v27EFFRQWmT5\/OOgppZKiYIISjGTNm1NnW2W\/evEFERAS8vb3h4+ODLl26YNu2bRg\/fjy8vb1RVVVVJ+etTa6urmjXrh3vRiVKSkpgY2ODqVOnflEfEUkTHx+PK1euwNraGnJycqzjkEaGiglCONq\/fz9iY2Pr5NiOjo7Q1taGoaEhjIyMcPv2bTg7OyMxMRHGxsaYNGkS7t+\/Xyfnrg0pKSm4cuUKrKysePeB5ubmBmlpaaxatYp1FLEcOnQIRkZG+O6771hHIY0QFROEcDRkyBCoq6vX+nEfPnyIkJAQWFhYVN+mqqqK1atXIyIiAmZmZrhx4wb09fVhaWmJxMTEWs\/wtfbv349u3bphyJAhrKNwkpGRUd0cjI+XggYHB+Phw4cwNTVlHYU0UlRMEMJR\/\/790bx581o9ZmVlJaysrDBx4kT06dPns\/t1dXXh4uKCO3fu4Pvvv4e9vT2GDBnCuTV3XQoNDcXNmzexYcMG1lE4O3DgAFq2bMnL5mAlJSXYvn07pk2bxtu9Wwj\/UTFBCEd2dnZ4+vRprR7T19cXOTk5NUYl\/s7w4cNx8eJFuLq6QlVVFevWrcOYMWMQEBDAdD3Fhw8f8Ouvv2L+\/Pno3r07sxziePDgAc6dO4dt27ZBSkqKdRzOvL29ISsrixUrVrCOQhoxKiYI4Wjp0qXo2bNnrR2vqKgI7u7u2LRpE1RVVf\/z8QKBAKamprh79y5sbW0RHR2NSZMmYe7cuczWU1y8eBH5+flYuXIlk\/OLSygUVu+\/oaenxzoOZzk5OfD29samTZt424+ENAxUTBDCUXx8PPLy8mrteC4uLpCTk4OhoSGn56mrq2Pz5s24d+8e5s2bBx8fHxgYGODnn39GZmZmreX7L7m5uTh9+jRWr17Nu\/UGV69eRWpqKiwtLVlHEYu9vT1atGiBsWPHso5CGjkqJgjhKCEhAVlZWbVyrLi4uOoPYnGvfujduzc8PT0REBCA4cOHY\/fu3RgxYgSOHj1aJ\/0w\/tf+\/fshJyfHu9bTJSUlOHLkCJYsWcK7Igj4uGD38uXL2LRpE2\/3biENB7XTJv+PvfOOr+n+\/\/jz7pu9hyS2kNhq\/bS21ihqtqVGq4sqRVu0RqsDFSIVm2itIKFmURqbGNHYe5NY2fvOc35\/XPKtUlyS3ETP8\/HIo3rvOZ\/P6577OffzPp\/Pe0hYSU5ODnK5vECqYA4YMACNRsO0adMKQJnFd2H58uV8\/fXX3L17l9q1azNx4sRCK7R17Ngx3nnnHRYsWFDiIjgmTZrEpk2b2Lp1KxqNxtZyrKZHjx6UL1+eiRMn2lqKhIS0MiEhYS2LFi167kJfYCmEdfbs2Sc6XVqDnZ0d77\/\/PkeOHGHo0KGcPXuW9u3b079\/f44dO1Zg\/dxn3rx5dOrUqcQZEleuXGHFihWMHj26RBoSf\/75J4mJiSW2PLrEi4dkTEhIWIkois+9rGw0Gvnxxx9p27YtZcqUKSBl\/8PPz4+wsLB8f4p58+bRuHFjvv76a9LS0gqkj507d3L06NESGUUwc+ZMGjduTOvWrW0txWry8vKYPHkyvXr1wsfHx9ZyJCQAyZiQkLCafv36PXc67QULFnDnzh0++OCDAlL1aGrXrs2iRYtYuXIlgYGB\/PTTT7zyyissW7YMo9H4zO1mZ2czduxY3njjDUqXLl2AigufAwcOsHfvXgYOHGhrKc\/ErFmzyMnJ4b333rO1FAmJfCRjQkLCSkJDQ4mPj3\/m8+\/evcuyZcsYN24cXl5eBajs3+nevTvbt29n9uzZ5Obm0qtXLzp06MCuXbueqb2FCxei1WpL3ISs1+v54osvaNmyZYnLhwFw6dIlFixYwHfffYdWq7W1HAmJfCRjQkLCSjp27PhcWxOzZs2iXLlydOjQoQBVPRlXV1cGDBhAbGwsn3zyCX\/++SctWrRgyJAhXL169anbuX37NqtWrWLUqFElLgoiKioKQRAYNmyYraU8EwsWLODll1\/m1VdftbUUCYkHkIwJCQkrycvLQy5\/tlvn5MmTREdH27SGgp+fH7NmzWLv3r107dqV8PBw6tWrx5QpU56q1PnkyZPx9vamefPmhS+2AMnOzmbZsmWMHDmyyFaECpLDhw+zffv2EpmuXOLFRzImJCSsJCoqiosXL1p9niAIfPvtt7Rq1YpXXnmlEJRZx8svv8yqVatYsWIFvr6+DB8+nMaNG\/Pbb7\/xbxHjcXFxrFu3jpEjR5a41NNTpkzB3t6+yFeECgK9Xs+oUaNo2rQpVatWtbUcCYmHkIwJCQkrGTZsGDVq1LD6vI0bN3Lz5k1Gjx5dCKqenbfffpt9+\/YxYcIE7t69S\/fu3enVqxf79u174Diz2cyUKVPo2bNniUs9ffr0aSIjIxk2bBhKpdLWcqxm9erV6HQ6vvrqK1tLkZB4JJIxISFhJXFxcdy+fduqcwwGA\/PmzePjjz\/G19e3kJQ9Oy4uLnz99df5UQ7Lly+nVatWDB48mMTERAB+\/\/13EhMT+fLLL22s1npCQkLo1KkTTZo0sbUUq8nOzuaXX35hyJAhuLu721qOhMQjkYwJCQkriY+Pt7o2x6+\/\/oper+ett94qJFUFQ1BQEDNnzuTPP\/+kSZMmzJgxg+bNmzNjxgwiIiL47LPPcHFxsbVMq9i1axcXL14s0ORgRcns2bPRarUlcntG4r+DZExISFiJtVVDr127Rnh4OEOGDMHBwaEQlRUcr776KuvWrSMyMhJXV1cGDx5MTEwMgiDYWppV6HQ6Jk+eTLdu3ShVqpSt5VjN\/dotX3755RMzdWZkZBAeHs6ZM2ceeP3ChQvMmjWL3NxcMjMzmTZtGjdv3vzXdgwGA1OnTmXJkiUF8hkk\/htIxoSEhJXMnj2b48ePP\/Xx4eHh1KxZk\/bt2xeiqoLH3t6ed955h\/nz51OuXDns7Ozo2bMnvXv35siRI7aW91REREQUSXKwwmLatGm0atWKZs2aPfHYzMxM5s6d+1Cq9ytXrhAREUFeXh5ZWVnMnTv3sdt0Fy9eZMKECXz99dfcunXruT+DxH8DyZiQkLCSli1b4ufn91THHj16lLi4OH788cdCVlV4REVF0bVrV3bu3En\/\/v2JjIykUaNGjBo16rFPuLYmMTGRJUuW8M033+Ds7GxrOVZzP1350zpdymQy7OzsHnIwVSgU2NnZIZPJ8o95XGjzhg0baN68OaVLl+a33357rs8g8d9BMiYkJKykTp06eHh4PPE4k8nE999\/T\/PmzalYsWIRKCt4Dh8+zI4dO+jfvz81a9Zkzpw5\/PHHH9StW5eJEyfSuHFjlixZgslksrXUh5g3bx6VK1emY8eOtpZiNSaTiQkTJtC5c2f8\/f2f+rx\/C9f9++uPC+lNS0tj9erV9O3bl549e7Jx40b0ev3TC5f4zyIZExISVjJx4kSOHj36xOOWLl3KqVOnSmQhLLAUIxs9ejSvvfYalStXzn+9TZs2bN26ldmzZwPQt29f2rdvz9atW20l9SFOnjzJ1q1bS2yCp4ULF5KdnV3k6co3bNhAdnY2r7\/+Ou+++y63b98mLi6uSDVIlEwkY0JCwko++ugjqlev\/thj0tPTWbx4MePGjSuRjn8AkZGRZGVlPTL1tIODAwMGDGDv3r0MHz6cbdu20aZNG95++23Onz9vA7X\/QxAERo8eTcOGDalTp45NtTwLt2\/fZs6cOYwePRpXV9enPu9+orF\/bmHI5XJEUfzXRGR\/Z9OmTVy5coXOnTvTo0cPTp06xcqVK637ABL\/SSRjQkLCSgwGAzqd7rHHLF68GC8vr2IfCvpv3E89PWLEiMfmNvDz8yMkJIRDhw7RvXt3oqOjadq0KVOmTLGZ896aNWu4ePFiiU3wNH\/+fIKDg6122NVqtej1enJych54PT09ndzc3CcWBjt58iSnT59myJAhNG3alJYtW9K7d282btyYn2tEQuLfkIwJCQkriYmJ4fr16\/\/6\/vnz55k3bx6ffvopCoWiCJUVHCEhIbi6uj71hPbSSy8RHR3N+vXrKV++PMOHD6dFixb88ssvRRpOqtfr+fXXXxkxYkSxTA72JE6cOMG6desYOnSo1ed6enpSoUIFZs+enZ8HJT09ndmzZ1OlShUcHBwwGo2YzWbs7e0fOn\/jxo0EBAQwceJERowYwfDhw5k+fTpgKTAmIfE4ZOLTrH1JSEjkk56ejkKh+NeKme+99x5KpZKIiIgiVlYwnDhxgldffZWoqKhnKuaVnZ3NtGnTCA8P5+7du3Ts2JERI0bQuHHjghf7D2bMmMGmTZtYs2bNE\/MyFDcEQaB79+74+Pjk+6NYy6lTpxg4cCByuZzSpUuTmJiIKIrMmjWLoKAgEhIS6NChAwEBAXh4eORvffTo0YOIiAhee+21h3x8xowZQ0xMDFu3bi2RUTESRYNkTEhIWMmMGTNo2rQpNWvWfOi9Xbt28fXXX7NixYrnKlNuK0RRpF+\/fnh6ejJlypTnauvChQvMmzePmTNnYjKZ6NGjB99++22hRbacP3+ezp07M3XqVNq2bVsofRQm69atY9KkSaxateqpQ48fRVZWFrt27eLu3bv4+PjQvHnz\/GRpOp2OAwcOcP36dURRRCaTYTabqVu3LhkZGVSvXh03N7cH2ktNTeWvv\/6iUaNGODo6PtdnlHhxkYwJCQkrCQ0NpUWLFrz00ksPvdelSxeaN2\/OkCFDbKDs+dmxYwffffcd0dHReHt7F0ibBw8eZNSoUWzfvh1XV1e+++473nvvvQJ\/yh04cCAmk4l58+YVaLtFgcFgoFu3bnTr1o333nvP1nIkJKxG8pmQkLCSDz54n8DAwIdeX7hwISkpKSV2MtDpdEyaNImuXbsWmCEB0LBhQzZv3szy5cupWLEiQ4YMoUWLFqxevbrA\/Cni4+M5efIkI0eOLJD2ipoFCxZgNptLrMOuhIRkTEhIWMm0aVM5ffrEA6\/dvXOH0NBQBg0aVOIKYd0nIiKCjIyMQjGG1Go1PXr0YOfOnUycOJFr167RrVs3Onfu\/Nypuc1mS3KwV199tUQmB7ty5Qo\/\/\/wzH3744SMdIyUkSgLKJx\/yYmEyQ0qqifQMM3q9gEwmw8FehrubEleX4u95n5srkJxqIjtHQBBEFHIZTk5yPD2UaDXF3zZMTTORmm5GpxeRASoVuLko8fRQ8pjEfMUCg1EkJdWMwVyWxDuOnLtoRKsR8fFWM3PWTGrWrEb37t1tLfNfyc4RSE4xkpMrYDKDWiXDxVmOj5eapKRbREZGMnr06EJ1snN0dOSrr76ic+fOhISE8Ouvv7J582ZGjhzJBx98QPny5f\/13ORUE2lpJgxGy\/+r1TL8\/TRELY\/k+vUrxTriQG8QSEo2kZUtIAiADJwdlXh5KJk5cyZNmzaha9eutpYpIfHM\/Kd8Jq7dMHD0ZC7JKRZDQhBEQIZSCfZ2CsoEqHiptgPOjsVvUjYaRY6dyuXCJT1Z2WaMRgCLfrVahquzgqDKWqpVsUNeDG2i9HQzcUdzuHnbSG6egGAGUQS5HOzs5Ph4KalTw55SvipbS30kFy7rOH4yj7QMAZNZhmAWEEURBwd7UpNP88u8wSxaOJu6dYtfkqQ8ncCxk3lcvKInJ+d\/Y0cmA41GQYCfK6tXjkIhS2L27F+KVNv27duZMGEC27Zto0yZMnzyyScMHjz4geqqd5NM\/HUsh9t3Teh0AmazRb9Go0EwJfNzaC9GDP+cfu8Vvy0CUYQz5\/M4fU5HWroZg8Hycysi4uzkSOL1Q6xaMYoVKxYTGFjJxmolJJ4dxbhx48bZWkRhI4oQfyyXnfuySEk1Iwggk3Gv8I3lfZ1e4OZtI4m3jLi5KnB2Kj4zckammZ17szh6Mo+8PAFRlD2g32yGzCwz124YyMw2U8pHhUpVfB7zz1\/UEbMri+sJRkwmERn39VveNxpFklJMXL2uR6mQ4+OlKjarFGazSOyhHGIPZpOeKSAKIANkMhG5XI4gyFj862hcXMtS86U+uLqAg33xMUZT00xs253FqbN56PUiIjLk+WNHhkKh5a+\/DhC9YgbvfziW6tXK8JgaUAVO+fLl6dmzJ+XKlSM2NpaVK1eyb98+\/PxK4e9fjsvX5MTsSiPxthGzScwf8zKZDLXGgd\/XzSQ7K4umLUeg0cjwdC8+961OL7B7XzaH4nPIyhIQRfLHvUKhxGA0MWfGp\/j61aJqzbdxcQb7YjR2JCSs4T9hTByKz+FgfA6iCCrl\/yayv\/\/J5TKUShlZWQI3bxvw9lLh5Gj7H6bsHIGYXZlcvmZArZKhUDxav0Jh+ceduyYyMs2UDlCjUtp+Rr50Vc+O3dnk5Aqo1TLk8kfrVypkGIwi1xMNaDVyfL1tv0JhFkT27s\/m2Mk8ZDLL+Pi7kaPVOrB\/byTxcWvo2ScEvdGZm7fy8C+lws7O9pNCarqZrTsyuXnLiFolf2jsKBQKBMFEdORwypavTpVqfcjKMuLvp7KMpyJCoVBQp04dunXrhpubG+vXryciYj5Hj54lKc0bZ5fyaNRyRNGcr12jsSMx4RSbN0ymQ6cvcXKpwNVr2Tg5WrbMbI3RKLJ9dzZnzutQKP43du7\/ae0c2btzERfP7+WtXhPJytFyJ0lPuTIa1Grb37cSEtZi+1+8QubyVT2Hj+aAKEMhf\/JNqlbLSM8Q2H8oG6PRtjtAIpYVlRuJRtSqByeyRyGXWfbBL1\/Vc\/JMXpFofBw5uQIH4nLQ6YWnWilRKmSIAhz6K4dbt41FoPDxnD2v59ipPORy2UNP6wqFiqysFPbuXkKjxj3xL10NGXruJhuJjcvBbLbt2BEEiPsrh9t3jajVjx47KpWWI4c3kJZ2i3bthwICp87mcO7i41OFFxb+\/v6MGTOGPXv20L5DN37\/fSUTvmvD+jXfkZaagEbjgEKhuldrQuD3tZPwD6hG1eotEEUdeqNIbFw2qWm2r2B6\/FQe5y\/pUKkeHjtKpZq0lET27V5KqzYDKVUqEIXcyO07RuLuPfRISJQ0XmhjwmAUOXE6D7MJrMlqrFbLSLxl5MRp207IKSkmzl3UoZA\/2ZC4z\/0nnzPndGRlmwtX4BM4djKPpBSTVVsuSqWMXJ3AqbN5iEWXhfkhdDqR0+fyQMYjl\/3VajviDqzCzs6ZFq99jF6XZXldJefKVT0XL9u2bPPN2wYuX9WjUj76FlcqVeTkpBO7J5Kmzd\/D06scglmPCJw+m5e\/t28LgoOD+fKrJXw4cCmVKjVgw5qfmDKhPTti5mEw5OHo5MFfhzdwK\/EsHToNByzJtlRKGVlZZk6dtY0xdJ+MTDNnzufd2056+H2VWsu+PZG4upWiwf91R6fLBixj\/8x5HQmJhiJWLCHx\/LzQxkRWloykZBPyp1iR+DsyLKsCV6\/b9qa+cFlPbq7ZKkMILD9KaelmEm\/a7uk+TydyPcGA4hlGmFwON28b0dnw8qekiSSnmFA+YrlfpdJy69Y5du34lVea9sbe3hWz2fI0LJNZVgUuXbVMzLbiwiU9eoP4r\/4PSqWGfbuXgExGw0ZvYjDkWl5XyEhKMXHrju0M0TwdXE8wUL9hZwZ\/HsXbvX7CYMhj6cJhTJvSlfi4DcTtX0WT5v3wL10Ng+F\/Rr9Mdm\/s6Gx39W\/dNpKRaX7kVpFabcf1a8c5tD+aZi3fR63WIgiWay2XW7ZHLl0tOEP07\/71he1r\/6ztF5bGJ7X1vP0KgsCUKVMYMmQIO3bsIDMz87naK+ko3nzzzXH79++ndOnSHDhwgHPnzmFnZ8fmzZtRq9VkZmbyxx9\/4Ovry+XLlzl+\/DgBAQFs3bqV9PR0HB0dWbduHWazmZSUFHbu3Imfnx+HDh3izJkz+Pr6smPHDnQ6HUqlktWrV+Pj48OtW7fYsmULPj4+7Nq1i4SEBNzc3NiwYQMmkwlRFNmyZQt+fn4cP36cuLg4ypYty86dO7ly5QpOTk6sWrUKFxcXkpOT2bx5Mz4+Ppw6dYqTJ0\/i6+tNzLZjZOa4I5OJVjv0yeVycnNzuXb5AAEB7mzYsJG0tDScnZ1ZuXIlgiBgNptZtWoVTk5OxMfHc\/z4ccqUKcOmTZu4c+cOGo2GqKgovL29uX37NmvXrqV8+fLEx8ezb98+XF1d2bBhAwaDAaVSybJly\/D09OTu3Tvs3r2HpDQf9HrFMznECQLY26sQjFeJiorK1xgfH0+ZMmWIjo7O\/zxr1qzBzc2N69evs2HDBqpUqcK+ffvYuXMnPj4+REVFIQgCJpOJyMhIPDw8SEhIYOfOnVSqVIl169Zx48YNSpUqxZIlS8jKTEeh9OTYyWwUCuudKeVyOSaTyF+H\/yAnOxEHByciIyPx9\/fnyJEjbNmyhcqVK7N9+3YOHz6Mv78\/8+fPx8nJiezsbBYuXIivry8XLlxg586dVKhQIT+hlJeXFxERESiVSu7evctvv\/1GUFAQu3fvZuvWrQQHB7NmzUpOnslErvRHEMwP6JfJ5MgVSn5bMRa1xp7XO36BYDY99OMhk8koX1rOmjXRnDp1Gnd3d+bMmYO3tzcZGRnMnz+foKAg\/vrrLw4ePEhAQACLFi0iKSkJV1dXZs6ciaOjIydPnuT333+nWrVq\/PHHHxw4cICqVauyZMkSRFEkNzeXefPm4efnx7lz51i+PJIyZSpw7pKGXN2jDWmV2o4b10+wfMlwOnQeQdlytTEadfd0g0qp5crl02z\/czHBVasRGxvLsWPHCAgIYNasWWRnZ+Pg4EB4eDgajSb\/O6lWrRrR0dEcP36cihUrEhkZiVKpJCsrixkzZlClShWOHDnCihUr8PHxITIykqSkJNzd3Zk8eTJOTk6YjDqWLtuKUfAHUUSpUlOlalNq1GqNQqnk+NEt7Nu9lLzcdJq3+hBvnwqIovi3CVlBekYmSnkSKclXGTfue5ydndm\/fz8LFy6kUqVKzJo1i5MnT+Li4sKPP\/6IVqslPT2d77\/\/nuDgYLZs2cKvv\/5KcHBwfk4MjUbD999\/j6OjI0lJSUyePJmaNWuyatUq4uLiqFy5MtOnh3PzZiJOrtW5eiPvIWNCdu9G\/i3qW5ydvWndbjAmk\/6BsSOXKzCZdGzdvICLF87i4+vLtGnT8PT05OTJk\/m6tm3bxqZNmwgMDCQ0NBSz2YxcLmfSpEm4urpy48YNli9fTuXKlVm4cCFnzpyhbNmy\/PzzzxiNRjIzM5k+fTpBQUHExcWxfPlyatasybJly7h69Sr29vaEhYVhb29Peno64eHhBAYGsnr1auLi4qhSpQpz587l1q1buLm5MXPmTDw8PDh16hSRkZFUrlyZtWvXsn\/\/fjw9PQkNDcXOzg5RFJk6dSoVKlTg9OnTREVFUa5cOVasWMHp06fxvfd5ZTIZN2\/eZPHixfm\/Rxs3bqR69eosW7aMAwcO4OnpybRp03BycuLmzZssWLCAsmXLsm\/fPmJjYwkMDGTmzJnk5ubi4ODA1KlT0Wg0XLt2jWXLlhEUFMTatWs5ePAgpUuXZsGCBQiCQF5eHjNnzqRixYqcOHGCpUuXEhwczJIlS7hx4wbu7u7MnTuX9PR0VqxYwbJly1i0aBHLli0jNjYWlUpF+fLlUSpt77tTlCjT09O5ceMGgiBw9+5dsrOzqVSpEjdu3CAwMBC1Wk1CQgJ6vZ6srCySkpIQRZHbt2+jUqkwmUzcunULPz8\/VCoViYmJmEwmUlNTyc7Ozjcy3NzcMBqN3L59G4PBQF5eXv6\/09LS0Ov1mM1mkpKS8Pf3x2AwkJycjNlsJicnh7S0NARBICMj495kYyItLQ2dTofZbCYtLQ2TyURubi6ZmZmIgpnMTD2CILP6yR4sk4HRKJCZrcsfYHl5efl69Ho9JpMJnc7yvtFozP\/3\/fdEUUSn02EymfKPMZvNGAwGZDIZcrkco9GIyWTCbDZjMlmebs1mEzk5BowyMzK5Ep7hGVcmg8wsE0YvAYVCgVwuz\/+v2WxGoVCgVCrzdYqiJTpBpVIhCAJyuRy1Wo1cLs8vXSwIAlqtFoVCgdlsRqlUIggCGo0GtVqN2WxGq9Wi1WowmMj3Xn8WBAGUCgccHRwxm804OjoiCAJ2dna4uroiiiJqtRpnZ2cUCgXu7u7519Td3R2lUolCocDBwQGz2Yy7uzsODg6YTCZcXFyws7NDLpfj4uKC2WzGwcEBDw8PBEHA0dGB1EwV4j8MCbAsUV84u4+EG6d494PpKJWa\/Kf6v119TCaRnBwTTk6OKJUaVCoVnp6e+d+Dl5eXJSJBrcbOzg5BEPDw8Mj\/nF5eXtjZ2eHs7Jyvy8nJ6V4EiZBfZEylUuHj45P\/ffn6+iKICvJ0Zh516WUyOYgiO7bOJbDKy9R6qT36f+g3mwXMgppSpXzzv3+DwbJM5Ovri4ODA6IoUqpUKRwcHHBzcyMrKwuz2YybmxsKhQJRFHFyckKhUKDRaPD39wfAzs6OUqVKodVq8fX1xdHREVEUCQgIwNHRAUEU0WpdMCMHRMxmI3m5mfiUqsQ7faZQsdL\/8cvcj0lLu8Ws8N40a9GP19oNxsXFB6MxD7NZQESBXGmHs4ucGjVq4O7ujiiKBAUFYW9vT8WKFXF0dMTBwYGgoCA8PDxwcHCgWrVq2Nvb4+fnR1BQEFqtlipVquDn54ejoyNVq1bF3d0dV1dXgoOD0Wg0lC9fHlEU0Wq1VKtWDXd3V7KyDI+8ZdUqLcePbuH61WMMHBJ5b5w\/uJcnIiKKCspXqIy3lwPOzs7UrFkTLy8vRFGkVq1aODk5Ub58eVQqFRqNhurVq1O2bFmcnJyoXbt2\/tiqWbMmTk5OBAYGYm9vj4ODAzVq1MDHxwcnJyfq1q2Li4sLpUqVonr16tjb21OlShWcnZ1xcnKiVq1a+Pr6otVqqVOnTv61k8vlaDQagoKC8PT0zO\/Xx8cHURSpWbMmzs7OVKpUidzcXBwdHalTpw5+fn44OTlRp04kPsWGAAAgAElEQVQdnJ2dEQSB+vXr4+bmRmBgYP54r1WrVn5btWvXxsnJiYCAgPx+g4OD89utXbs2vr6++dfG2dmZwMBAPDw80Gq1VK9eHX9\/\/\/zP6+XlhSAI+e0GBgaSl5eHs7MztWvXxt\/fHzs7O+rUqYOjoyMBAQHUrl0brVZLUFAQbm5u+Z\/3\/u+MVqvF29ub1q1b06BBg\/x587\/GC51n4vQ5I1u3p6HRWD+jmc0i9nYKunb0wNVGCQ2j1qRx567xmcI8TSaRMgFqOr3uWgjKnsyNBCNbtmeiNwhWr6wIomWrqX0bL8oGFIq8J3IwXk\/swfQHEoHJ5QpkMjm\/zuuPp3d5ur75LTpdDv+cOcxmERdnBW+87oaLU9HvJOoNEBmdTG6e8I+nYxGN1onzZ\/fyW9Q3vPfhTEr5VUGvf9CY0OkEqgU70rqFA7bgeiJs3JKEIFqcisFi3Gs09qxe+R0pyTeoVbsdm34P5frVY5QpW5O27YdRo3Zr7OxcEQQdr7d2wd\/XNtFYMTszOX1O98B9K5crERGYP7MfFSrWp32nEeTlZT50rskk4uut4s3Obg+9J1H8WLduHS4uLtSvX\/+B3Cj\/RV7odRitRkSlkj3TE7IIqDWgVAjYyrXEzk72zPvuggDubrb7eu3sLCtC4r3EWlYhgkIpQyk3ArYJEbXXCqiUD44drdaR3Tt+5ebN83R+81tMJgOPegQVRdBoZGhtFOKnVIo4OSrIyX3wqVehUKHXZbN5wxSCqzW3GBKGh52MFQoZapXtvF8VMgMKBQhG8oeOWm3HlcvxnD21m559p1A56GUqB73MgdiV\/LExjHmz+lG1WgtebTuERi+3xdXZdmHdj8pRo9bYsSNmHtnZqbzctPcDfh4PcG\/smO9lt5Uo3nTq1MnWEooNL7QDppurHCdHOWbhGZxrzODuorRpEplypTXWTsOAZTJTKKB0gO2W2lxdFLi5Ku5lK7QOQRBxdZHj6mq7CcHLU4lWK7+XJdUyEScnX2f7n3Np0qwPPj4V7xkTD2M2g6e76plWxAoChVxGmdJqTP+49iq1Hft2LyU56QbNW36AWbi3F\/U37mcl9fWxXZ4PNzclzo6K\/PvW4kdgZOvm6ZSvWJey5WqTnZWKo5Mn7ToM46tv\/qThK29z+tQOwkO7EL3sSxITLttMv7eXCrVKxv0dDJVKQ\/KdK2zdNINGr\/TEza0UZvOjnaMF0XLtJUNCoqTxQhsTLs4KAvzVWFuYUBBApZYRXMWuSLMB\/pNKFTS4uyoxmawzhkwmEV8fFaVsmPhJqZRRpZLWEhljpS0nCBZDypaZJD3dlZTyVeUbQ0qlmn27l+LuEUDjZn3R63MeeZ7ZLGJvLye4srYI1T5MpfIanJ3k+fkuFEoNyUlXORgbTfs3vsTN3R+T8eEJTRAs6c0D\/Gw3duzt5FSqoOGeTyVqtT2HD64m4cZJmrf6EFEUEEUBk8lAXl4m3t4V+ODjuQwbsZYatduy5rcZtGzZlPHjx5ORkVHk+kv5qPDyVGIyW5a1ZDI5O7ZHULpMdRq90uOhbaX7GE0iTo5yKle07diRkHgWXmhjQi6XUSPYDicH+VMnoBJFy01duYKGcmVt60RjbyenZnU7uBdu+DSYBVAqoVZ1O5s9Gd+nciUNFctrMBrFp96uMRhEvL2UBFex7Q+qUimjTg177LQy5HItd+9cJD5uHc1bfYhG45AfPfB3RNFy\/asHa21eY8TDXUm1IDsEAURRhlql5sDeKNw8\/KnXsOu9ZfaHvxWTWaRsaQ0O9rbN\/lq9qj0B\/irMgpKMjNvs2vELLV\/tj59\/UH7kyX0MhjwEwUzl4Ff5YUIUvy5cgkajZsyYMTRp0oSNGzcWWKnzp0GjkVG7hgNqpSVdecKNk5w7vZu2HYahUts9cuwI99K0v1TLvkQUHHzRSElJYcaMGXzyySdcvmxZ1TIajQwcOJDVq1fbWF3J4IU2JsCyXN2ssRMO9vInJuIRBMtkVq6Mmob1HZ9pi6GgqRqkpU4NewRBfGJWRZNJBEGkTi17Kpaz\/dONQiGjUQNHfLyVGAziY1coRNFy7V1dFDR72alY1Ebx81XRuJEzWo2MjRvC8C9dnaDgpo+I3rBsbRiNIkGBWurWKh5lpGvXsCeosha5XMOVy0eJO7SaJs3fQ6FUPTShiSLo9SJeHgrqVLezeW0UrUZGs5edKO3vzIF9q7Czc6HhK28\/0tdAFEGnM+HsqKdJIyfee7c3sbH7+fHHH7l79y4dOnTgnXfe4dChQ0Wmv2J5NY0aOCOTGVm\/JoQKlRpQtlytR+o3my33ds1qdtSoaldkGiX+R2ZmJgqFgujoaGJiYgBLrojU1FQOHz5sY3Ulg\/9EbQ53VyUe7kpS081kZpsRxftL77J7MeqWyUAulxFcWUvzxk44OhQPO0sul1HaX41CKeNusgm9wfKULwKIMgThf\/odHBS80tCRurUcbD4Z3MdOK8ffT01enkBqmulvPhQW50ZBEDELFkPO309Fy6ZONn+q\/ztenkpOndjG4sVz6NF7Mg5O3piMBkCGIIqI9669UimjZjV7mvyfI5piUgpeqZRRtrQaUSYnPOwLnFxK82qbgRj0Osu1v6\/\/3vUv5auiRWNnvL2Kh1+2g70cXe41Jk78ipavfUJA6ZoYDJZVCfHe2L+vvXxZNS2aOOHhZrn2jo6ONG3alC5dupCTk8OSJUtYvHgxqamp+SGThY2vj5ID+9YRHbWQt3tNwsHR856fjWXsCGbL2FGrZdSr48D\/1XMs0pooEv\/Dzc2N+vXrc+jQIcxmM23atEGhUODt7Y23tzeVK1e2tcRizwsdGvpPcnIFzpzTkXDTUl3TYLBUIbSzk+HmoqRSBTWVymuLzUT8T27dNnL2go6kFBM5uQImkyVaxd5OTqVyGsqWUePhXjwmgn9iNoucvaDjyjUDmVlmcvMsy85ajQxnJwVlS2uoEqhFa+OtmX9iMBjo3u0N6tZrRp\/3vubYiXSyc0UMegGZ3BK14ummJLCSlnKli2ds+Y4dMYwYMYovv16EWl2OtIycfANIo5bh5CgnwE9NUKC2WBQo+zuffjqAO3dS+HFiJJeu6ElLN6LXC4iAViPH1VlB2TIW7Y8Lod6yZQtTpkwhJiaGsmXLMmbMGHr06IGjo2Ohac\/JyabXO2\/RpPkbtGo9gIuX0sjOETAaRZRKsNcq8PJUUiVQg3+p4jl2\/mt89NFHGI1GFi5cSHJyMitXrqRr1674+PjYWlqx5z9lTNzHYBDJyTVjNFn2KTVaOU4O8mJrRPyT3FyBPJ2A2QwKJdhr5cVuEvg3zAJk55jR6y3DTq2S4WAvL1Yl0\/9OdHQ08+bNY\/36tdjbO6LXiySnZGM2y7C316LVyovNKtajEEWRzp070bp1az79dBA5uQJ5eQKCAHKF5frb28lRFoMKs\/9k3759DBs2jOXLl1OxYkVLIrBcAf297UqN2jJ2nla7Tqdj9erVfPPNN1y6dImXX36Zzz\/\/nG7duhWK\/oiICH7\/\/XdWroxGpVKTpxPIzRPyawVptfJiVa5eAj7\/\/HMuXrzI+vXrWbJkCUFBQdSvX9\/WskoE\/8mRrFbLcHNV4u2pxMtTibNjyTEkAOzt5Xi4K\/H2UuLhpiwxhgSAQg4uTgq8PS3X39VFUWwNiZSUFMLCwvjww4+wt7c8wWo0MubNncStm8fx9FAWa0MCYNGiReTk5NKnT1\/AsnXg6WEZO57uSpydFMXSkACYMmUKb7zxBhUrVgQsKykuzv8bOy7O1mnXarW88847HDhwgDFjxvDXX3\/RvXt3PvroI06fPl2g2hMTEwkPD+fjjz9GpbKsOthp5Xi43btv3ZWSIVEMcXJyIisri4MHD+Lr6ysZElYgjWYJiX9h0qRJeHt7061b1wdeb9CgET4+fjZS9fQkJCQQGhrKgAEDcHZ2trUcq1i5ciV3797lo48+KvC2PT09+eGHH4iNjeXdd98lIiKCl19+mbFjx3Lr1q0C6WP8+PHUqlWLtm3bFkh7EkWDr68vR48e5cyZMzRt2tTWckoUkjEhIfEITp06xa5du\/jhhx9QqR50CPXx8cHevnhEbDyOn376ieDgYLp3725rKVaRlpbGhAkT+OCDDwp1r\/qll15i4cKFbN68mXLlyvHjjz\/SpEkTVqxYgU737GXMDx06RGxsLN988w1yWyaqkbAaT09PmjVrRseOHdFoNLaWU6KQRrqExCOYOXMmrVq1ombNmg+9t2TJEk6dOmUDVU\/PmTNniI+PZ\/To0baWYjWzZs2ibNmy9O3bt0j6a9u2Lbt37yY8PByZTEbPnj3p0qVLfoigtUyfPp23336bwMDAAlYqUZgkJycjCAKzZs3Cw8PD1nJKHP9JB0wJicfxxx9\/MHLkSNavX0\/ZsmUfev\/27dtoNBrc3IpvMaZ3330Xf39\/JkyYYGspVnHlyhV69epFeHg49erVK\/L+7\/s6TJ48GZlMRp8+fRg7dmy+38aTiI6OZurUqfz+++94enoWslqJ50UQBFasWIHZbCY1NZXXX39dMgKfEWllooSi1+u5detWkWb2K0iysrJISEiwtYyHyM7OZvz48fTt2\/eRhgTAkSNHbJKm+WmJjo7m8OHDDBw40NZSrGbKlCnUqVPHJoYEgL+\/P5MmTWLPnj107tyZRYsWUa9ePSZOnEhqaupjz01LSyMsLIx+\/fpJhkQJIScnh8WLF7NmzRratWsnGRLPgWRMlFBSU1PZtm0bJpPJ1lKeievXr\/P777\/bWsZDLF26FHd3dwYNGvSvx+zYsYPExMQiVPX0ZGZmMmfOHIYPH05AgI3qtz8jW7du5c8\/\/2TIkCG2lsIrr7xCdHQ0a9asoXz58owaNYrGjRuzZMmSfz1nzpw5eHp60q9fvyJUKvE8ODo6snTpUpYsWSIlpnpOJGOihGI2m9Hr9baW8cyYzebncnIrDFJSUli2bBkDBw58rPPVmDFjqFOnThEqe3pWrFiBu7s7ffr0sbUUq8jLy2PatGkMHDiw2PyoKxQKOnfuzJ49ewgJCSE1NZW+ffvStWvXh1JzJyQksHHjRkaMGIFaLSWgKinIZDI8PT1xcHCwtZQSj2RMlFBkMlmJ9hQvjvq\/++47AgICaN269WOPW7p0KefOnSsiVU\/PlStXmDt3Lh9++CEKhe1rm1jDmjVrkMlkfPzxx7aW8hAODg4MHz6cvXv3MnToUNauXUujRo345JNPuHbtGmAJI65VqxZNmjSxsVoJCdtQvH7NJSRsxMGDB\/MdL2VPyGBWqlSpYhk2NmHCBCpXrkybNm1sLcUqbt68ycSJE+nXr1+xDrmtVKkSYWFhxMbG0qZNG+bMmUPTpk3p378\/27dv56uvvrK1RAkJmyEZExL\/ecxmM6GhofTr149atWo98fjatWvj51e8klbt3buX48eP8\/333z\/RGCpuTJ06lYCAAN544w1bS3kq\/u\/\/\/o9169axdOlSvLy8mDdvHomJiUValVRCorghGRMShUpJiDzeuHEjly5deupsi9OmTePEiROFrMo6Jk2aRJcuXUqcN\/rx48fZsWMHoaGhDyUHK86oVCp69erFF198gaurKzKZjO7du9OlS5cCT80tIVESkIyJF4yYmJhiNdEZDAY2bdpEcnKyraU8kvT0dEJDQ+nfv\/9Th\/P16NGD8uXLF7KypycqKork5GTef\/99W0uxmp9\/\/pkOHTpQtWpVW0uxmjt37jB37lwiIiKIjY2lX79+bNiwgXr16jFixIh8fwoJif8CkjFRgpHJZPme4+np6fzwww+MHz++WC3BazQa9u3bR5cuXdi9e3f+68XFQXD69Omo1Wqrsi3a2dkVG+fRlJQUpkyZwieffIK3t7et5VjFhg0bOHHiRLF0unwaQkJCcHFxoUuXLgQHB\/PLL78QExNDvXr1mDx5Ms2aNWP69OklOupKQuJpKR6\/iBLPTGJiIosXL6Z169Z88803VKtWDRcXF\/R6fbH569q1KwcOHKBDhw589tlnHDt2DL1eb\/MJ+X6p4TFjxqDVap\/6vIiICC5cuFCIyp6eWbNmUaZMGd555x1bS7GKzMxMxo8fT+\/evfH397e1HKs5deoUe\/bsYdy4cQ+M4+bNmxMTE8PMmTPRarV89tlnvP7662zdutWGaiUkCh8pnXYJJSUlhS+++IL9+\/dz\/vz5\/NcrV66Mt7c3ZrPZhur+h1wuRxRFjh8\/TnZ2NmAp81utWjV69uzJZ599ZjNtX375JWazmbCwMKvOu3LlCh4eHjavxHn16lXeeecdpk2bVuJKJf\/8889s27aN6Oho7OzsbC3Havr374+Hh8dj05XfvHmTiIgIQkJCyM3NpUePHnz++ec2y+4pIVGYKG0tQOLZMJlMBAcHExwczKpVqzh8+DAATZs25Z133sFoNBYL50elUkl6ejoDBw4kOzsbNzc3unXrRrVq1Wy6MrF37162bt3KqlWrrD539erVtGzZ0uaJq0JCQqhbt26JMyRu3bpFVFQUEydOLJGGxObNm4mNjX1iBlc\/Pz+++eYbunbtyg8\/\/MDy5ctZtWoV48aN4+OPP5ZSbku8WIgSJZKEhARxyZIloiiKYlpamjh\/\/nyxSpUqYrt27Wys7GE2bNggymQysU+fPuKhQ4dEURTFy5cvi9OmTbOJntzcXLF169biuHHjnun8WbNmifHx8QWsyjo2b94sBgYGihcuXLCpjmfh888\/F99\/\/31REARbS7GazMxMsUWLFmJ4eLhV5xkMBnHdunVis2bNRECsUqWKOGfOHFGn0xWSUgmJokXymSjBGI1GDAYDrq6ufPjhh2zdupUqVapw5MgRW0vLx2g0cuTIEZYsWcLixYvzn6Lvb3nYgtWrVyOKIl988cUznd+3b1+Cg4MLWNXTo9PpCAsLY9CgQVSqVMlmOp6F2NhY1q9fz7Bhw0pcPgyw+MvI5XI++eQTq85TqVS88cYbbN26ldmzZ5OamsqAAQNo3749Bw4cKCS1EhJFh2RMvECUKVOGiRMnUqZMGVtLeYDPPvuMXr162VoGAElJSSxcuJBPPvkER0fHZ2pj8uTJHD16tICVPT0rV65EpVKVuCgIk8lEWFgYvXr1onr16raWYzUpKSlERUXx+eefo1Q+2w6xWq1mwIAB7N+\/n6FDh7J\/\/36aNGnCoEGDOHPmTAErlpAoOiRj4gVDq9Xi4eHxXG3k5eWxdu1a4uLinluPSqXCxcXludspKH766ScEQaBDhw7P3Ebt2rXx8vIqQFVPT3JyMnPmzKFfv35WRaAUBzZu3EhaWhrDhg2ztZRn4ocffqBy5cq0bdv2uduqWLEiYWFh7Nmzh3bt2jFz5kxatGjBhAkTyMnJKQC1EhJFi2RMSDxEfHw8gwcP5tSpU7aWUqAcO3aMLVu28O233z5XtsVGjRrZzHlu4sSJuLm50aVLF5v0\/6wkJyczdepUevfuXayMy6clNjaWjRs38vnnnxeo4\/BLL73E6tWrWbp0KX5+fowePZrGjRuzcuVKjEZjgfUjIVHYSMaExENUrVqVxo0b06BBA1tLKVBmzJhB586dadq06XO1M3v2bJtkGT169Ci7d+9m4sSJNs\/RYS2hoaEYDAZ69OhhaylWYzKZ+Pnnn3n\/\/fepXbt2gbevVCrp1asXf\/75JyEhIdy4cYO33nqLTp06sWvXrgLvT0KiMChZv0gShUpGRgaXL1\/mr7\/+Qq1WU7ZsWVtLKjB27NjByZMn+fTTT5+7rX79+lG5cuUCUGUd06dPp0OHDtSoUaPI+34ezp07R0xMDFOmTClxWzMAW7ZsITk5mYEDBxZqPx4eHgwfPpxDhw7lO1Q3b96cL7\/8UkrNLVHskYwJCcCyjBsaGkpcXBzh4eGUL18eBwcHW8sqEHJzc\/n+++\/p2bMnpUqVeu72jh49WuT72mvWrOHgwYMlsv7G7NmzadGiBa+88oqtpVhNZmYmISEhRbo9U6FCBebPn8+ff\/5Jx44dCQ0N5ZVXXmHKlCk2jYKSkHgckjEhwenTp5k0aRIdOnSgW7duZGZmPlUp7pJCREQEBoOBDz74oEDaO3DgALdu3SqQtp6GjIwMQkJCGDBgAKVLly6yfguCLVu2sG3bNvr3729rKc9EWFgYcrmcnj17FnnfLVq0YO3atSxbtgwXFxeGDx9Os2bNpNTcEsUSyZgooYiiiMlkKpC2fvnlF5o1a0aDBg2Ii4vDaDTyf\/\/3fwXS9r8hCEKB6X8ciYmJrFixgnHjxhXYSsvw4cOLdKth4cKFeHt7P3WJ9OKCTqdj0qRJ9OzZk4oVK9pajtVcuHCBP\/74gwkTJtgsU+d9QyYmJobvv\/+eK1eu0KZNG3r06FEg0VYSEgWFZEyUUJRKJa6urs+d+Eev13Ps2DFatmyJ2WzO3+Lw9fUtIKWPRqvV4u7uXqh9AMydO5eqVavy2muvFVibS5cuLbJCX7dv32bp0qV88cUXaDSaIumzoIiMjESpVDJ06FBbS3kmZs2aRYMGDWjUqJGtpVCqVCnGjh3L4cOH6dy5M1FRUTRv3pyRI0eSkpJia3kSEpIxUVLx8fGhW7duzxXiCJYnHy8vL3bt2kV8fDx6vR4\/P79Cz05YuXJlq8p+Pwvx8fGsW7euQJwu\/44oikVWQn3SpEnUqFHjuSNQipr09HSWLl3K4MGDsbe3t7Ucq9m1axc7d+5kwIABtpbyABUqVGDNmjVs2rSJunXrEhISQoMGDZg\/f77kTyFhU6SqoRJcunSJa9euUaNGDdLT05HJZCUuTfM\/EQSBTp06ERwcTEhISIG2ferUKUqVKlXoKyv79u3jo48+Ijo6usRljBw9ejQXLlxg+fLlRWZ4FRR6vZ727dvTsmVLRo0aZWs5\/4pOp2P58uWMHTuWxMREmjVrxldffVUgSbUkJKxFWpmQoGLFirRs2RIvLy8CAwNLvCEBsH79epKTk\/nyyy8LvO1Zs2Zx7NixAm\/37xiNRkJDQ+ndu3eJMySOHz9OVFQUX3zxRYkzJACioqIAiv32jFarpV+\/fhw8eJBhw4Zx5MgR2rVrx8CBAzl9+rSt5Un8x5CMCYkXjry8PCIiIhg0aBDe3t4F3v4HH3xAlSpVCrzdv7Nx40YyMjKsLihla0RRJCQkhLfeeouGDRvaWo7VpKenM3v2bD799NMSsz3j7+\/P1KlT2b59O717984PxR01ahSJiYm2lifxH0EyJiReOEJDQ9Hr9XTv3r1Q2re3t0etVhdK2wBpaWlMmzaN9957Dzc3t0LrpzDYvn07ly9fLrH1NyZPnoyHh8dz1W6xFXXr1mXJkiVs3LiRgIAAJk6cSIsWLYiKipJSc0sUOpIxIfFCcf78eRYuXMiwYcMKLfph1qxZhbqM\/PPPPyOKIm+99Vah9VEY5ObmEhISQt++fW1WCO15OHbsGBs2bGDUqFHP7dhsS15\/\/XU2b95MWFgYZrOZHj160KFDB7Zs2WJraRIvMJIDpsQLxdChQ5HJZISFhRVaH7dv38bR0fGZS5g\/jvPnz9OrVy\/Cw8OLRUiiNfz000+sX7+emJiYErNFcB9RFOnTpw\/+\/v5MmjTJ1nIKjMTEREJDQ\/Pvh\/79+zNq1CjKlCnzxHMzMjLIycnBz8+vsGVKvABIKxMSLwwHDhzg8OHDDBo0qFD7iY+PL7QMmPf3u0uaIZGQkMDy5csZM2ZMiTMkAHbv3s3169cZMmSIraUUKPf9KXbs2EHnzp2ZO3cuDRs2JCwsjNTU1Meeu3nzZqZNm1ZESiVKOpIxIfFCoNfrGTVqFO3atSv0bIs3btwgPT29wNvdsGEDMTExJTL19Jw5c2jUqBGvv\/66raVYTV5eHuPGjePNN998YZ\/CmzdvzqpVq1i\/fj2Ojo58\/vnnvPbaa\/mRK48iJiaGlStXkpSUVIRKJUoqkjEh8UIQGRlJTk5OkUQ\/vP322wQHBxdom3l5eUyaNImOHTuWuNTTBw8eZNOmTYVeVbOwmDdvHtnZ2fTr18\/WUgoVhUJBx44dOXDgAOPHj+fGjRv06NGDN998k8OHDz90\/KVLl7hy5Qq\/\/fabDdQWLTk5OfkPCPv27WP9+vWF1ldsbCynTp3CaDSyYsUKrly5UmBtZ2VlkZubC1icoTdv3lxgbT8JyZiQKPGkpaWxdOlSRo8eXSQpuqdNm1bgDpjLli3D2dmZ0aNHF2i7hY3BYGDcuHG0a9eOmjVr2lqO1SQkJBAdHc2ECRMKxQemOOLh4cGoUaPYt28fgwYNYtWqVTRr1owhQ4Zw+fJlwJIQ6\/7kOnXq1Bc+xHTixImsW7cOsDgSZ2VlFVpf8+fPz+8rKysLg8FQIO2KosjIkSM5e\/YsYDGQirK6sWRMSJR4Zs+ejYeHBx07diyS\/ipWrFigE09KSgqRkZEMGjSoxJV9X7t2Lbm5uQwfPtzWUp6JefPmFXjtlpJCYGAg06dPZ9euXTRo0IDw8HDq16\/PsmXLOHz4MAkJCYCl4NnTZpHNy8sjIyMDnU7HpUuXHtgOzM7OJicnh6SkJK5fv57\/elJSEhcuXHhg4svMzCQ3N5ecnBwuXbqU\/7R9H5PJxNWrV7l06RJ6vf6B93Jycrh48SK5ubnk5uY+kGY8PT2dCxcuPGAcZWRksG3bNk6ePEl2djbNmjWja9eu+e\/r9XouXrz4gJ+U2WwmNTUVg8HAjRs38q\/V467LpUuXyMvLw8HBAZlMhkqlolevXlSqVAm9Xp+\/OnL16tX8IogZGRlcuHCBtLS0h9pMTk7m4sWL+Z\/\/8uXLbNmyhbNnz2IymWjTpg3t27fPP95gMHDp0iVu3rz5wGupqakYjUauX7\/+XEaj8pnPlJAoBpw8eZKoqCjmzJlT6PVE7tOiRYsCdTIMCQnB09OzxKVB1ul0LFy4kIEDB+Lq6mprOVbz119\/sW7dOhYvXmxrKTaladOm\/PHHH0RGRhIWFpY\/wf3dQXPx4sW8+eabNPxJXDwAABucSURBVG7c+LFt7dixg2nTplGuXDmuXLmCyWTiq6++onXr1kRHR7Nx40Z0Oh1eXl5Mnz6dBQsW8Ntvv2FnZ4dcLmfUqFE0bdqURYsWcejQIRQKBQkJCdjZ2TFhwgRq1KjB6dOnGT9+PElJSej1etzd3fnpp5+oUqUK8fHxjBo1CrPZjJeXFwaDgbZt2\/Lhhx+yatUqIiIiUCgU3L59m\/bt2zN27Fj27NnDhQsXSE1NpUWLFly9epUzZ84wffp0Tpw4wbhx4\/K3D1q1asWYMWPQ6XS89dZblCpViszMTBITE2nbti1jx459KCT9xIkTfP311+h0Onx9fbl69SpBQUHodDo++ugjhg4dil6vZ+7cufkGVUREBAcOHCA8PByFQoHJZOLjjz+mW7dugCU8feXKlcjlcpydnRk1ahTr1q3j7t27zJkzh5dffjnf3yUkJIS4uDjGjx9PdnY2Op2ODh068NVXX3Ht2jUGDx6Mr68vqamp3Lp1i06dOjFmzBjrB5IoIVFCEQRB7NGjh\/jpp58Wab9DhgwRd+7cWSBtxcfHi1WqVBHj4uIKpL2iZNy4cWLbtm3FvLw8W0uxGqPRKHbo0EEcOnSoraUUK3JycsSRI0eKwEN\/r7\/+upibm\/vY86OiokRA\/Oabb8TExERx3LhxYq1atcSkpCQxNDRU1Gg04tq1a8UbN26Iy5cvF\/39\/cWoqCjx5s2b4ldffSXWq1dPTE9PF8ePHy8C4qJFi8Tr16+Lffr0Ebt27SoajUZx5MiRYr9+\/cTk5GTx+vXrYv369cUxY8aIOp1ObNeunfj++++LCQkJ4qJFi0SFQiH+9NNPYlZWlti2bVvxl19+EQ0Gg7hu3TrRz89P3Ldvn6jX68VWrVqJ3377rWg0GsURI0aIb7zxhpiTkyO2atVK7Nmzp5iYmChu375drFChghgWFibm5uaKKpVKfPfdd8Vr166JCxcuFF1dXR\/6XcjNzRU7duwo9u7dW7x27Zq4bNky0dXVVQwPDxezs7PF+vXri9u2bRO3bNkiKhQKccaMGeK1a9fEEydOiA0bNhR\/+eUXMS8vT4yOjhYbNmwoXrx4UdyzZ49YpkwZcfny5eLVq1fFN998U\/zoo4\/EGzduiHXq1BHXrl0rms1msX\/\/\/mKfPn3ElJQUsU6dOmKvXr3E69evi1u2bBHLly8vLl++XLxw4YJob28v9u7dW7x+\/boYEREh+vj4iJcuXbJ67EgrExIllpiYGK5evcqUKVOKtN9u3boViJOkKIr8\/PPPdOvWjXr16hWAsqLj7NmzLFu2jNmzZ6PVam0tx2p+\/\/13cnJyGDt2rK2lFCuuXbvGxYsXUSgUmM3mB97btGkTs2bN4osvvvjX800mE9WrV2fIkCG4u7szePBgfvvtN\/bs2YOdnR3Vq1enU6dOgKUya5cuXfKTs40aNYqYmBj2798PQPfu3fMrCw8fPpx3332Xq1ev8vXXX3Pt2jUOHDjArVu3EAQBnU7HtWvXSElJYc6cOfj7+9O3b19WrVqFXq\/H3t6eOXPmkJyczJo1a7hw4QKiKJKRkYFarcbe3h5XV1eUSiUqlQpnZ2dOnDjBnTt3CA8Px8\/PDz8\/P3r16sWmTZt48803KfP\/7d17VJR1\/gfwN+AMIMggeQGBSUtzyUsHRMVITemYu6KZR9PUTNYN77WaqWxpeYsMEIybgAni8bKGUgqEWEZ6NI0VDETTZEFYZEdRhuUizADz+8N1frJa+nDxO0+8X\/94zjDP8MbD8Hzm+\/le1GrMnTsXarUa48ePh5OTU7MWAgAUFRWhrKwMYWFhUKvVUKvVOHjwoLE1YWVlBQsLC+j1ejg7O+P111+Hg4MDEhMTUVZWhtzcXOTk5MDc3By\/\/PILTp06hatXr8Ld3R0zZswAAOzbtw8VFRXo3LkzFAoFunfvDnNzcygUCtja2uKnn36CVqvFxx9\/DFdXV7i6uuKVV15BYmIiBg8ejO7du2P+\/PlwdXXF+PHjERYWhqKiIjz11FOSfndYTJAs6fV6REVFGTcaepxKSkrQs2fPVr\/OsWPHcPnyZQQFBbVBqscrNDQUkyZNwtixY0VHkay6uhpRUVGYN2\/eY5mwKxcnTpzA3LlzjZMwH2Tjxo3w9PTE6NGjH\/j1pqYm2NnZGduACoUCCoUCt2\/fhpmZWbP2oF6vb7ZTqkKhgLW1NaqqqmAwGJq9xywtLWFhYYH6+nqkp6dj8+bNcHJygoeHB7p06YJOnTqhrq4OSqUS1tbWxuueeOIJWFhYQKfT4dNPP0VmZiY8PT3h5OQEOzs7Y2u0qakJhnv2bzQzM0N1dbWx0Lira9euaGhogE6ng0KhMF6j0+ke2GbV6XT3vUb37t3ve67BYICNjQ3Mze9MY6yqqkK3bt0wePBg6HQ6dOrUCVFRUXj++ecREhLSbM6Wubk5zM3NodPpYDAY0NTU1Oy1b9++DYVCgS5duhgfs7OzQ01NDZqamqBQKIx57v4chhbsZckJmCRLO3bswK1btzB79uzH\/r0vXLjQ6rX3NTU1CAwMxJw5c9rlMLL2dPr0aeTm5rb75mDtZevWraipqWk2ya4jq66uRlxcHCZNmvSbhQRwZwLjX\/\/611+dcGhpaYnz58\/jxIkTAO7MoaipqcHQoUNRV1cHg8FgvFH17dsXKSkpxtfKyMjA9evX4e7uDoVCgcOHDxuXTaakpMDe3h4qlQpBQUHw8\/NDamoqFi1ahLq6OtTV1UGtVkOv1yM1NRUAUFhYiG+\/\/Ra2trbIycnBkSNHkJSUhJ07d2L06NG4deuW8eat1+uNkx6BOzfVAQMGQK\/X48svvwRwZ9XYV199hT\/84Q9QqVRobGw0\/ix3\/\/3fIsHZ2RkNDQ3Yv38\/gDsjFd99953x+97r3kKgT58+0Ov18PLywvz58zFjxgzcvHkTSqUSHh4eyMnJQVFREQBgw4YNmDZtGurq6poVEgaDAY2Njejfvz\/q6uqwe\/duAHcmfB89ehSjRo2CQqFAY2Oj8bpf+zkeBUcmSHYKCwsRHByM4OBg2NnZPfbvv3LlylYfrR0bG4v6+nq8+eabbZTq8dDr9diwYQMmTZqEJ598UnQcyS5duoR9+\/YhLCys2SfYjuz48eM4ffo05s2bB0dHRzg4OKBTpzu3Bp1Oh9raWmi1Wty8eRNFRUUoKipCXFwc1q1bd99rmZmZQalU4uOPP8Ynn3wCjUaDxYsXo1+\/fqioqEBlZSUMBgPMzMwwd+5c\/Pjjjxg\/fjwcHR3x73\/\/G\/7+\/ujbt6\/xYLLFixcbl2quX78ezs7O8Pb2RkREBL7++mvY29ujtrYWxcXFsLe3x7Jly7B27VokJibCxsYGnTp1gqWlJVxdXdGrVy\/MnTsXzs7OsLa2hqWlpfGGPGDAAAQGBsLR0RH19fWoqKiAo6MjVq5cicDAQBw+fBharRbdunXDqlWrANy5Kd\/N2dTUBK1We9\/Kkm7duuG9997DqlWrcPDgQdjY2KCiosJYVFVWVkKv10Ov10Or1RpbS+PGjUNmZiZeffVVqNVqaDQaDBw4EK+\/\/jqmTZuGY8eOwdfXF927d0dZWRnWrl2LHj16QKVS4S9\/+YuxcCgvL8fTTz+N1atXY+vWrUhOTkZFRQXUajWWLVuGf\/3rX6ioqLjv52jJwXA8m4Nk5\/3338cvv\/xirPYftx07dmDYsGEYOHBgi64vLS3Fq6++ik8++UR2bYLt27dj27ZtOHbsmJBCrrWWLl0Kg8GAiIgI0VFMxt2bu5Tnl5eXP\/Awt927dyMyMhKbNm1CVlYWPD09jb\/j2dnZuHr1KiZPntxsWP3AgQMoLi7GyJEj8fzzzwMAAgICUFFRgRkzZuDMmTPN9jGpq6vDnj17UF5ejokTJ8LW1hZnz56Fj48PSktLUVZWhsuXL+OFF17Ahg0b4O3tjaVLl6KgoAAHDx6Eg4MDXnvtNeTm5sLc3BwjRozAjRs38Pe\/\/x0DBgyAra0tbt26hZdffhkAkJOTg4yMDDg6OmLq1KmwsbFBbW0tkpOT8eKLL8LZ2RlVVVVIS0vDkCFD0Ldv3\/v+X3788Ud89913GDVqFADAxsYGbm5uSE1NxbBhw9DY2IiTJ09i8uTJzeYgHT16FNnZ2VCr1Zg6darxADq9Xo+kpCQUFxfjT3\/6EwYNGgTgzqjp119\/DV9fX1RWVqKqqgo+Pj4AgHPnzuHIkSNwdHTE9OnTYWVlhRs3buDIkSPw8fExrkxJT0+Ht7e39Pax5CmbRALl5uYavL29Dfn5+cIyhIaGGs6dO9fi6wMCAgz+\/v5tmOjxKC0tNbi7uxuSk5NFR2mR06dPG0aMGGEoKioSHeV3KyEhwTBgwABDTU1Nq15n+fLlhpkzZ0q6RqvVGl566SXD\/PnzDZmZmYaNGzca3NzcDGfPnm1VFno0nDNBstHU1IS1a9fCy8sLzz77rLAc\/v7+eOaZZ1p0bVZWFo4cOYKlS5e2car2l5CQgL59+2Ly5Mmio0jW2NiINWvWwNfXV5btGbno0aMHhgwZ0mz+QUv06dMH\/fv3l3SNSqVCYGAgysvL8dZbb+GHH35AbGwsPDw8WpWFHg3bHCQb+\/btQ3BwMFJTU9tkNUVLhYSEwNvbG15eXpKu0+l0GDduHIYMGYKQkJB2Stc+8vLy4Ofnh6ioKAwbNkx0HMkSEhKwbds2pKWlcQVHB9DY2NjqeU0kDSdgkixUV1dj586dePfdd4UWEgAwcuTIFmVITk5GQ0MDAgIC2iFV+2lsbMS6deswbNgwWRYSN2\/eRExMDFavXs1CooNgIfH4sZggWUhMTIRSqcTUqVNFR0Hnzp0ln81RW1uLhIQEvP322+jWrVs7JWsf6enpKCsrQ2RkpOgoLRIfH48+ffpg0qRJoqMQ\/W5xzgSZvIKCAoSHh2PJkiXG2cwiRUVFIS8vT9I1mzdvhoWFhexuaHq9HnFxcZg3b57wEaGWOH\/+PBITE7F06dIHru0norbBkQkyeUFBQRg+fLjJnOy4fPlySQdbXbx4EXv37kVsbKzstp6Oi4tDZWUlpk2bJjqKZAaDAQEBAfD09MSIESNExyH6XWMxQSYtMzMT2dnZOHDggOgoRleuXEG\/fv0euV3x2WefYeLEiXjxxRfbN1gbu3r1KmJiYrBu3bpmW\/HKxdGjR3Hr1i1ER0eLjkL0u8dxPzJZt2\/fRnBwMHx9feHq6io6jlFmZiY0Gs0jPfeHH35AXl4eli1b1s6p2l54eDiGDx8uy6Wg957d4uLiIjoO0e8eRybIZF2\/fh39+vXD\/PnzRUdpZs6cOXB0dHzo83Q6HdatWwdfX1\/Z3dCys7ORmZlp3JZXbiIiIlBVVYVZs2aJjkLUIXBkgkyWq6srgoKCTG7i3969e3H58uWHPm\/nzp2orKzEwoULH0OqtnN3g6exY8dK3jjIFBQXF2PHjh1YtmyZLNszRHLEkQkyWXeP1jU1gwcPfugEzNLSUsTGxmLt2rVQqVSPKVnb+OKLL1BaWoqEhATRUVokNjYWXl5e8PX1FR2FqMNgMUEk0ZgxY6BUKn\/zOXf3NpDbDa2yshLR0dFYs2bNAw9yMnW5ubn45ptvsGPHDtFRiDoU0\/vYR2TiNm7ciHPnzv3q1\/Py8pCcnIwVK1ZIOo3RFMTHx8PBwQFTpkwRHUWyhoYGrF27FiNHjhR6dgtRR8SRCSKJZs6cid69ez\/wa42Njfjwww8xYsQI2W09feXKFezcuRNhYWGyK4KAO+2Za9eucSkokQAsJogksrW1RVNT0wO\/lpqairKyMsTGxj7mVK0XGBiIQYMGYfTo0aKjSFZVVYXt27djyZIlcHJyEh2HqMNhm4NIokOHDqGgoOC+x\/V6PT7\/\/HP4+\/vL7vyNzMxM5OfnY\/369aKjtMiuXbvQpUsXzJgxQ3QUog6JIxNEEi1YsACdOt3\/1omJiUFtbS1ee+01AalazmAwICIiAtOnT\/\/V9o0pKywsREREBCIiIh46MZaI2gdHJogkSk5ORklJSbPH\/vnPfyImJgZLliyBjY2NoGQtk5iYiOvXr+PNN98UHUWypqYmbNiwAUOHDsXYsWNFxyHqsDgyQSTR9evXUVNT0+yxyMhIjBw5Eq+88oqgVC1TVlaGkJAQrFixAg4ODqLjSHbq1ClkZWUhJSVFdBSiDo3FBJFEixcvbrbaITc3F2fOnJHl3gZxcXEYPHgw3njjDdFRJDMYDIiOjsacOXPw5JNPio5D1KGxzUEk0WeffYbz588D+P+tp0eNGoVnnnlGcDJprly5goyMDLz77ruyXAq6a9cuFBQUwM\/PT3QUog6PIxNEEjk6OhrPfNi\/fz+Ki4sRFxcnOJV0y5cvx6BBg+Du7i46imT3tmfktnKG6PeIxQSRRG5ubujZsyeqq6sRFxeH1atXo0ePHqJjSXLo0CEUFhYiNDRUdJQW+fzzz\/Hcc8\/xVFAiE8E2B5FE+\/fvR2lpKeLj42FlZSW7paB1dXWIjIzE8uXL8fTTT4uOI1lBQQFSUlLw9ttvm+RBcEQdEd+JRBKtXLkS1dXViI+Pl+V8g8TERCgUCsycOVN0lBb58MMPMXToUHh6eoqOQkT\/xTYHkUR5eXlYv349vLy84OPjIzqOJCUlJYiMjMSmTZtgaWkpOo5khw8fRl5eHlJTU0VHIaJ7cGSCSKLU1FT07t1blltPBwcHo3\/\/\/rI7Gh0AdDodtm\/fjoULF8LFxUV0HCK6B0cmiCT66KOPoFKpZNevz8rKwqlTp7Bnzx7RUVpk9+7dqK+vl+VOnUS\/d\/L6a0hkAtLS0oz7TMjF3fM3JkyYgH79+omOI5lGo8GWLVvg7+8Pa2tr0XGI6H+wmCCSSKlUPvCgL1OWlJSES5cuYeHChaKjtMjmzZsxePBgTJ48WXQUInoAef1FJDIBPj4+sLKyEh3jkWm1WgQHB8PPzw89e\/YUHUeyf\/zjHzh69CiSkpJk11oi6ij4ziSSaOPGjTh79qzoGI8sPj4eTk5OeOutt0RHaZGIiAj4+vqif\/\/+oqMQ0a9gMUEk0YwZM2Sz2dO1a9eQlJSE9957DxYWFqLjSLZ\/\/37k5OTgnXfeER2FiH4DiwkiiVQqlSzmTBgMBqxZswZubm7w9vYWHUeyiooKhIaGYsGCBXB0dBQdh4h+A4sJIonCw8ORn58vOsZDffvtt8jIyJDtp\/r4+Hi4uLjA399fdBQieggWE0QSzZo1C25ubqJj\/CadTofo6GisWLECgwYNEh1HsmvXruGrr77CokWLZNmeIepoWEwQSXTx4kVUVFSIjvGbDh06hKqqKvj5+YmO0iLr169H7969MWbMGNFRiOgRmH7jl8jE3LhxA9XV1aJj\/CqNRoOPPvoI77\/\/Puzs7ETHkSwzMxPff\/89vvjiC9FRiOgRmRkMBoPoEERyotVqYWFhgS5duoiO8kB\/+9vfkJ+fjwMHDshioui9GhoaMHv2bAwfPhzLli0THYeIHhHbHEQShYSEmOx22j\/\/\/DO++eYbBAYGyq6QAIDk5GRoNBr8+c9\/Fh2FiCRgMUEkkZubG7p16yY6xgOFhoZizJgxePbZZ0VHkez69esIDg7GnDlzoFKpRMchIgnk99GFSLCXXnoJCoVCdIz7pKSk4MyZMzh06JDoKC2ydetW9OrVC7NmzRIdhYgk4sgEkURxcXEmt89EdXU1tmzZgjfeeANqtVp0HMkuXbqEY8eOYc2aNVAqlaLjEJFEHJkgkmj69Omwt7cXHaOZXbt2QalUYtGiRaKjtMinn34KDw8PeHh4iI5CRC3AYoJIorKyMlhbW4uOYVReXo69e\/di1apVJpXrUWVkZODkyZNIS0sTHYWIWohtDiKJDh8+jKKiItExAAB6vR5r1qyBWq3GH\/\/4R9FxJKupqUFQUBAWLFiAp556SnQcImohjkwQSfTBBx+YzLLL\/\/znP6itrcU777wDc3P5fTbYs2cPzMzMsHjxYtFRiKgVuGkVkUS7d+\/Gc889h4EDB4qOAoPBgIaGBpNcXfIw5eXlmDZtGlasWIEJEyaIjkNErSC\/jzJEgl24cMFkttM2MzOTZSEBABs3bkTXrl3x8ssvi45CRK1kGmO1RDLi5+dnsptWyUVWVhbS09Oxa9cuk2kZEVHLcWSCSKLo6GiT3U5bLsLDwzF79mwMHTpUdBQiagMsJogkmjBhAlxcXETHkK309HRcvXoV\/v7+oqMQURthMUEkUc+ePWW5n4Mp0Gq1WLVqFaZPn44ePXqIjkNEbYTFBJFE27Ztw88\/\/yw6hizFxMTA3t4ec+fOFR2FiNoQZz4RSbRo0SJ+qm6BkpISpKWlYdOmTejcubPoOETUhjgyQSSRRqPBrVu3RMeQnaCgILi5ueGFF14QHYWI2hhHJogkys7Oho2NjegYsnL8+HGkp6fjyy+\/FB2FiNoBd8Akkqiurg4AYGVlJTiJPOh0OkyePBmjRo3C6tWrRcchonbANgeRRKGhodxnQoKDBw+ivr4eS5YsER2FiNoJiwkiiXr37o0uXbqIjiELWq0WsbGx8Pf3h62treg4RNROOGeCSKJx48bB0tJSdAxZCAkJgUKhwJQpU0RHIaJ2xJEJIomioqKQnZ0tOobJy8\/Px+HDh\/HBBx\/I9jAyIno0nIBJJNH58+dhZ2cHtVotOopJu9va2LJli+goRNTO2OYgkujGjRtwcHAQHcOknTx5EhcvXsTu3btFRyGix4BtDiKJDh06hCtXroiOYbJqamoQEBCAiRMncvSGqINgm4NIovLycigUCqhUKtFRTNLmzZuxb98+fP\/997CzsxMdh4geA45MEEmUkpKCkpIS0TFMklarxYULFxAYGMhCgqgD4ZwJIok0Gg3q6+tFxzBJtra2CAsLQ9euXUVHIaLHiG0OIolyc3Ph7OyMJ554QnQUIiKTwDYHkUTx8fHIz88XHYOIyGSwzUEk0ZQpU+Dq6io6BhGRyeDIBJFETk5OPJuDiOgeLCaIJAoLC+OpoURE9+AETCKJCgsL0bVrV9jb24uOQkRkEjgyQSRRSUkJtFqt6BhERCaDxQSRRD\/99BM0Go3oGEREJoNtDiKJampqYGFhASsrK9FRiIhMAkcmiCQKDw\/HhQsXRMcgIjIZLCaIJOrVqxeUSqXoGEREJoNtDiKJysvLoVQqeZAVEdF\/cWSCSKKgoCDk5uaKjkFEZDJYTBBJNHLkSLi4uIiOQURkMlhMEBERUauwmCCS6Pjx4ygpKREdg4jIZHACJpFEGo0GVlZWUKlUoqMQEZkEjkwQSXTixAmUlZWJjkFEZDI6iQ5AJDfu7u5cFkpEdA+2OYiIiKhV2OYgIiKiVmExQURERK3CYoKIiIhahcUEERERtQqLCSIiImoVFhNERETUKiwmiIiIqFVYTBAREVGrsJggIiKiVmExQURERK3CYoKIiIhahcUEERERtQqLCSIiImoVFhNERETUKv8H1S3x0lw\/Z5kAAAAASUVORK5CYII=","width":245}
%---
%[output:4f34d9b8]
%   data: {"dataType":"textualVariable","outputData":{"name":"target_azimuth","value":"90"}}
%---
%[output:88c31395]
%   data: {"dataType":"textualVariable","outputData":{"name":"target_elevation","value":"0"}}
%---
%[output:7bd987d2]
%   data: {"dataType":"textualVariable","outputData":{"name":"interf_azimuth","value":"38.3865"}}
%---
%[output:933c7c63]
%   data: {"dataType":"textualVariable","outputData":{"name":"interf_elevation","value":"0"}}
%---
%[output:851559af]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAR0AAACrCAYAAABbjkinAAAAAXNSR0IArs4c6QAAHPxJREFUeF7tnW+oF8X3xycQf4JKmmmKktceSCFSIWFioYLoN4z8hqGZhfpApYSUJP+G\/woUDFRKRH1wFQvrgo8LCq6iQtGDiggLhHsFw\/stuioV3VDw13v0fJy7d3Z3dndmd2b3bFyufT4zO7PnzL7umTPnzNx3+\/bt24IvlgBLgCVQkgTuY+iUJGluhiXAEpASYOjwQGAJsARKlQBDp1Rxc2MsAZYAQ4fHAEuAJVCqBBg6pYqbG2MJsAQYOjwGWAIsgVIlwNApVdzcGEuAJcDQ4THAEmAJlCoBhk6p4ubGWAIsAYYOjwGWAEugVAkwdEoVNzfGEmAJMHR4DNRKArdu3RKDBg2q1TPV7WEYOnXTaAOeB2DB1dfXJ3\/\/\/fffAp\/h56+\/\/hK\/\/\/67mDBhghg\/fjwDyMPxwNDxUCncJSEBEgcW+u7KlSsCP7\/88ov46quvZPmxY8eKF198UUJn8ODBYsSIEWL48OEMH48GFUPHI2U0sStkoeC3arGoYIFcvv76aykewAWgof+Pk9nmzZvFu+++K3777TdpEQE+I0eObKKIvXtmho53Kqlfh3RgUa0YQEQHFlgw9F0eqbS1tYkVK1aITZs2SfigHwyfPJK0W4ehY1eejb1bFCwQBD4jvwv+PzodoqlRFrAAJPQza9Ys+W9c3d3dYuXKla0pVk9PT0sXBJ8333xT\/Pnnn\/LzYcOGseVT0Whl6FQk+BCbVcFy8+bNlvM2DSyqFWPy3FGwoA4+mz17dmz148ePt6DT0dEhJk+eLNavXy\/OnDmjhc\/169eln4fhY6IRu2UYOnblGfzd4hy4SWAhKybNz6IKhywUAglZLUlgSRJuFDrz5s0T999\/v4QOLCBYQnSplg\/Dp\/why9ApX+aVt2i6MqRaKKYOXBOwkCVjUxBx0KE2du3aJVAmCp9t27aJRYsWCYIPnM2wfvhyJwGGjjvZVnrnImDJ6sDV+VlcgCWPpROtA\/js3Lmz38foaxQ+o0ePFkOGDKlUh3VtnKETuGbTlpxp6qPGs9hw4Jr4WcoUbZqlYwqfY8eOiWnTpknLB9ABfDjC2a4mGTp25enkbmlLzjqw5PWzqCtDvoHFhqVjCp8vvvhC+oSw2sXwsTusGTp25Zn7bqZLzuRnoeVnk0A5Ez9LXgdu7ge2XDGrpaM2Dz\/PiRMnBky7IBNYPkOHDm0FGHJ0c3HFMXSKy9D4DqZLzgQW\/PbdgWv88I4LFoEOdS0OPvABLVu2rB98OLo5v0KdQ+eHH36QS5a9vb2tXi5evFjs3bs3f689rmm65EzTH8obsuFnUR26HovISddsQMcEPhzdXFx9zqFz9OhR2cvVq1cX760ndzBZGSKoRB24qhVj8jh5AuVM7lu3Mjaho8KHltpVecHy4ejm\/CPIOXSQeDd37lz5E9KVFywEm7yBcoDMxIkTUyNwQ5JlGX11AR1T+HCAYTYNO4UOPP+AzsWLF0VXV5fsmW9TK9MlZ9VCselnCd2Bm224uSvtEjoqfOAq4NSKYnp0Cp1ff\/1VLF26VCxZskROrwChtWvXinHjxpXq0zFdctaBJcRAuWJDIszaZUAnDT5wNm\/YsIGjm1OGkFPo6NqGYxmJeAcOHBBTp061NsJNlpwR5IUVCtuBctE8ImsPxTcylkCZ0KFOxeV1ReHD0c391Vg6dGD9rFmzRuzevbsFnYMHD8ql4X379skd3+IukyVn8qmoYAFsYLGcP3\/eeBCzA9dYVF4UrAI69OAmeV0cYHhvmDiFzpdffin27NkjTp06JcaMGSNbhaWzfft2ceTIkdZnAATAc\/r0aTF9+nRZ58EHH5TlaTc5Ncs5ChbsnYJ7wIrJ68DFPYtmOnvx9jW0E1VCR4VPXF7XwoULObr5rqCcQifOpzNz5kztEjrA8fbbb8voz40bN8r9WmClXLhwQYaiw5y1FYFbdkJiQ1lQ2mP7AJ00+CC6ecqUKY2PbnYKHSiBwGO6eoVp1gsvvCDzXi5dumQ8aH3IdDbuLBe0LgGfoJMVPk2LbnYOnayjC9CZM2dOv20u6R7sZ8kqzeaU9xE6kH5SXtfhw4dlakXT9m4OBjq029uOHTua8ybxkxpLwFfo0AMk5XXB3zNq1ChZtAnbp3oNHXXTbdXawQ7\/DB\/j97ERBX2Hjgl8kFrRhOhmr6GDaN3Ozs7EfW4ZPo1gSupDhgIdho8QQUBHdczp9rkFeGD98NVcCYQGHRU+uqRSbJ9a1+jmoKCTtirQ3t6eeExJc1\/J+j95qNBR4aPL66pjdHOQ0GH41B8iWZ8wdOhkgU\/o0c1BQydtSRKWDzmjsw5iLh+WBOoCHZJ6Ul7XunXrgo5uDh46aY45OKMZPmEBJE9v6wYdkgGeCz6f6HldmHZhxwZE7+N89pD2bq4NdNLgg5yY5cuXs+WT540OoE5doZPmSoDDecGCBS34hBDdXDvomMCHl9kDoEjGLtYdOmnwQXQz8rpCiG6uLXTSliRh+TB8Mr7ZHhdvCnTS4LN\/\/355WCAuX6Obaw+dtFUBjm72mCQZutY06KQtogA+SJrGLg2+wacx0GH4ZHiDAyzaROikuRLwBxUrXb7BJxE62IQr6egYHC9j+5QHNcuc0iBcvANxS5Js+biQtvt7Nhk6afDxLbp5AHTU\/W+STm6gTdbPnTsnJk2a1G93wCJDrCzopMVDMHyKaLH8ugydezKPy2iPwqeqvZv7QQfAwc592KuYthc1GT5UD+dBF73Kho7qmOO8rqLaq64+Q2eg7AGftLyuKqKbG+fTSXstoCTdPrec15UmuWq\/Z+jEyx\/wicvrQoDhP\/\/8Iw9EgNO5jIuhEyNlhk8Zw89eGwyd\/tMrgAY\/Z8+ebUUzq4cEUmnE98yfP1+eRQerp4wrETrR\/Y2jHbLpy6F7VzW90gk7aatJTq0oY3iat9E06BBUCCyQFP6tA0uSFL2CDjmK0eFDhw7Jtf4yLp+gQ8\/L8ClD88XaqCN0KN+KQEJWSxawRA+CxDFLuDDdwuUVdMjK2bJli\/Vl8aTh5SN00uDDeV3FgGGjdqjQiQMLWTImsiGw0MEFdH5b3DFLuDdmKd5BhyyduDOqTISRp4zP0DGBD6dW5NF68Tq+Qyc6HVL\/3\/TpdccsoS7i2bJc3kIHD+Hq3PFQLZ1ov+OWJDmvK8srYKesD9Cx4Wcp45gl76GDuV9vb692ZNTdkWz6Oujgw0fmmErPTrmyoKOC5fLly9J5m9WBq\/Oz4LOsFkteyXkLHZpeYSlt7969eZ8vc70QpldxDxUXD8HRzZmHQeYKNqHjyoGrWjGZH9BiBW+hw47k\/FqGUnFKaXS3N4ZPfpmm1cwKHdJNNJ4liwMXfQrxOGtvocOO5LRhnv49J5Wmy8hWiTjo6By4aDPrsrO6MkSwKWs6ZEtGdB9vocOOZHuqpvyXqOXD53XZkXE0juq5554T165dE5iqm16qxYJ\/T5w4UVoxoYIl6bm9hU5aNDIeih3JpkP6TjlOrcgmL7W0Sz9LHcESJHTyD49iNUN2JJs+OcNHLykbgXK489ixY8Wjjz4qLZW0QDlTndWpnLeWTlVCbgJ0INvolIDk3YQjc1wEymH5mnYH6OjoEPPmzZM75vE1UAJeQafJ++lUNTjrCp+yHbhZV6+q0rcP7XoFHQikaTsH+jAIkiwfn\/O6bEXg4vnJr0LToax+FoaO+Uj2Djpq1+u8R7K5isotGWf5VJVa4dKBG5eQmEfiDB1zqXkNHfPHsFeyKT6dNIlhYOi2mnQBH1sO3CoD5Rg6aSPq3vcMnYisGDr9BaKDT968LhcOXPTWh3gWhg5Dx1wCDB0jWQEYun1uo6kVtvwsoUXgMnSMhpEsxJYOQ8d8tNwdMDr4ABJZQ\/tVB27brDYxsW2iWDF7Rab+5C7c3X2naltb7luoFRk65mJk6DB0zEeLUlKX1xW9UdwWCmTFdItucUKcEGfu\/kf120SbWCFWiB1iR66+xVYCaHBM0Zkzd35aDbYJsWKFEDvyt8fQMVcVQ4ehYz5aNCWRzU4WDqZaphG4AM4usUscF8dj258tZotO0Vmof63KAM6\/x\/uI4\/HtCeyA15mvPYaOuZq8gg5idNasWSN2794tpk6dmvgUWE7fs2ePtZM9qTF2JJsPHpRUUypu375tVBnAWSlWSvsm7bICHgAHG4Gr1k1cwwbg6evrk7Vv3LjR+v3JJ58I2ir2ww8\/FE8\/\/bSYMmVKaceqpMnRp++9g87SpUtFV1eXSDpSGAJk6PgxjPJAB9YNoKNOpzCVwrSKLCD8pqtdtMvpVu4L1s3d0wfkPeDHwVQKv8kCIh8Pvm9vF30vvzwALKqT\/LvvvpPfX716VdC\/qX8vv\/yywDgeOXKkePjhh8VDDz3E8FGU5yV0nn\/+efHxxx\/LbuJ8J53Vw9DJ\/QparZgHOnPEnJaVA9B0ia4BfVLLFLZ25sy5Z+UANF332oPVghMm\/+8\/\/xFD7m5F0ff002I36gghgaIDi4kQN2\/eLOEDiwjwwXYVfHm2ehXdLRBKQ\/Kczuph6PgxfLNCBxbMJHHn+BFc8NkAKtErag3dFmZTtwE3ggVz97gTfHf5xAnx89ixshj5ogCWx7\/9Vuzp6WlVv89AvLpgRJwNFT0Wev369eKVV14RABzDx3PoQO84EQJLtMjYPXXqlBgzZowcDgwdg7eihCKuoBOFE6whWEVxF1ks+E2+FpjxfT\/9JF5Yt65Vbem\/2058osCFvsCdVXsLWMQEL8+xK3HR3BjHr7\/+umwS47iplo+X0yvdAXtk9eD36tWrGTolAMWkiazQwT3vE\/fsiDh\/DZzMmGLRBUtHBQv9GwP4p59+Ej09PS3fSnQ6pNpI8CRF169kRDOm8opf50xnZ+Fd++ICKrHiBwDh7O4mwicY6EStnlWrVoljx47x6pUJGRyWyQMdTK\/IURz11wAmPUN6+q9udQvR8U2H+PbbbyVc8JPFzwILhmwkrJedUJb2ZTZ5dHUr4vcpKj6GT38JBgUd6jpZPbxdadHXoXj9PNDRrV7998Z\/xWP\/e0zCqP3hdgme1qUzTzRdjz3PiaDSmkvdDQbEudoxq1cyWNDypQuoxC6D2FOZLB9a7bLctFe38wo6WSQDX8\/27dvFkSNHWn6eLPXjynKcTjYpxkFHF8sifSx9fTL3Zs+8PaLnUQUscc3CsXLP7yxLAS7d3bMlF7q7Z4kdO9rkZ7BnEIoD4wW\/+3FDXcFKekTLVo6uKRP4TJ48ubY7DwYLnWyvhnlpho65rFBShc7PP\/\/cggoGFvlZMBXCFY1nkcHGCcdgw3E8e9dsMWvirBZsMB2isBsKswFk2tvvZDjs3HkPOvgsE3hwQ0QkW8rHSpOk7pQOWD5wNsPvg8UTwAe+nzpdDJ2INhk6A4d3dGVI+l16eqQD9\/PPPxefffaZrPTEE08MBEvC2yKtE6Q7wWJRggFNcq+ihgsBiJoDbACdARelQqjBgKhcMPeqCBR0G+UDPvj8scceqx18GDoMHSmBuCVnLD\/DgUuWShYHLk2F8DttO1CCTtKyePTFjpsxxQJHvYHlLPMi0KG6cfDBau6TTz5Zm+hmhk6DoBO35KwDC60SZXmZAJYyj10BNwCeqNGiBBxn6b4XZfFCnvh3nhgNMEQuF+BDqRUhx\/gwdGoGHZ0DF5\/B3wIrRfWzZAWLLlAOLwgyrHGZJnzaeLtplwr4cKKXQc6mjS44vUccfDCF3bp1qzxbK9To5tpDh85EP3funBwkaYmkIfh04sBCCYlJgXJpb0oULDRFijsdIc+SeVof0r5PAg7VrQN48Cxx8MESO5baoa\/QAgxrDx3E9ODau3evIADNnDlTRjXrLl+gYwKWPIFyWfwsaS8\/vq8COlgOx7SKnMfkwyEfD30+YPXK5IE8LZMEn9Cim2sNHcTyINnuwIEDrUx15GydPHlSHDp0SAwbNmzAECsbOrqVISgFmc8uHLg2j12pCjpoFzM6wIZ+kyLjPveUJZm7lZTXFQp8ag0dXVKoDkSq5l1AJ27JGQ5cF34W22BJejOqsHQyv6k1rKBLrQglurn20IlaNWm7E+aFjm7J2ZUDN83PUuY7xtApU9oD2zKBj2\/RzQwdIcSVK1fEhAkTpEaToKNbctaBBffJE8+iOnGxJOrDeU5prxRDJ01C5XyfllrhU3Rz7aET3UdZN706ePCgOH36tBg\/fryED\/a4BUwQG4ElYQILho8LP0vWc7PLGcZmrTB0zORUVqkQ4FNr6Jg6kmHdbNy4UVo8ZKXQ6hHiIrLEs1Cms3pYnGrBlDX4ymqHoVOWpLO1E5fXRUvtVe7dXGvoQE0mS+Zk6UyfPl1aN6hz6dKllpYHDRokbt261U\/reXaUyzZswijN0PFbTyapFWVHN9ceOibBgapPB0NIPcOIhhQggyNGQvCzlPkaMHTKlHb+tnyCT+2hk1dNungIWDzvvPNO63yjvPeuUz2GTjjaTEutKCu6maGTMmZ0S5JQDk6zpMPVwhl29nvK0LEvU9d3rBo+tYYOnSTR29vb0mNa7lWcwhk+eskwdFwjwt39q8rrqjV0jh49KjUWl2eVR526JckmWz4MnTyjyK86JnldNvdurjV0sAo1d+5c+WP70i1JkrMZU6+mXAyd+mjaJK\/LRnRzbaGDVStA5+LFi\/JsdFx5p1ZJw0q3KtAk+DB06gMdepK01Iqi0c21hQ4dUbxkyRI5vaKl83HjxsltLmxfcfDBWewhRxynyYmhkyahcL93BZ\/aQken6rQMcxvDIw4+nZ2dd49JsdGKP\/dg6PijC1c9SUutyBrd3CjopGWY21JanGMOFg8sH0qVsNVelfdh6FQp\/XLbNoGPSXRzbaETt5eOiwP64lQfBx9sur18+fJawIehU+6L70NrJnldSfCpLXTifDpJW5W6UmgcfN577z2xbds2V82Wcl+GTili9rKRpNSKGTNmxO7dXFvoQEsEHperV1lGg7okiZQKOLVxbdiwQaxbty7Lrbwpy9DxRhWVdSQrfGoNncq0kNKwCp8RI0bIs4yQxR4ifBg6vo6ycvuVJbWCoVOubvq1pi5JEnywidiiRYuCsXwYOhUOIA+bNoHP4MGD5RHUuA4fPizmz58vrf6yzmy\/73aZJ7R5qCR0ieBz\/vx5eUIFLB\/AZ+HChdL68fli6Pisner6lpRaAeCQO4GhU52OZMu0JIn9fVT4LFiwQGzatKni3umbZ+h4qRZvOhUHH+ogjohas2aNXMllS6dCtWETMbzMKnxwvMgbb7whXn311Qp7NrBpho5X6vCuM4AOfs6ePTvgfHZ0dvTo0fIPLHbt3LdvX+uABJcPwtOrBOnSC42VLky5oBzf4MPQcfl6hHFvAgvBhVwGsNzTLqQkzZo1Sx6MgJ8y4MPQSdOKcnQv4IO\/DDBDkXT3wQcfiGeeecbgDu6KMHTcydanO6tguXz5srRe8GMCFnoOisSnXETARt2RAZY99iuHP9NlCAlDJ8PIUi0fFT779++vLKmUoZNBgZ4XBUTIt4jfmBLZAItvp6EwdDIORNUxB4sH8IEF9PjjjwvAp+y8LoZORgVWXDwOLGS5mHZPdxpKmcdZm\/ZTV46hk1N6KnwQ4zN8+HB5JyyxY6m9LPgwdHIq0HG1qJ9F\/X\/TpqNgQb06nIbC0DEdATHlovApO7qZoVNQgQWqF3Hgqn6WEI+zLiA2wdApIj2lblXwYehYUmDCHxXys9h04IZksdg8YAGyZOhYHrOU1\/XRRx+1AgyR14XVABfRzQyd4gq06cCt43HWtg9YYOgUH7PaO8SlVtiObmbomCvQhZ\/Ft5Uhc2mYl7R9wAJDx1z2uUrq4IMAQ8DHxl4+DJ3+atGBhaZHpgqsqwPX9PnVci4OWGDo5NFEjjpqXpca3VwUPk2Eji0HLtSoBsqF5GfJMQRzVXFxwAJDJ5cq8ldS4aMGGL700ku5LJ+6Qsemn4XBkn+86moWPWCBoWNXH8Z3o31uEXquwgcO5yyHBYYMHQ6UMx4uXhUsesACQ6didRI01Ohm5HWZplaEAB1XDlx1elSxGmvbvIsDFhg6ngwXgoca3Txq1KhU+PgCHVt+FnXJGaphP0u1A9TFAQsMnWp12q\/1uADDadOmxeZ1lQkddTrU1EA5j4ZLaV2xfcACQ6c01Zk3FAcf2rtZzeuyDR2XDtxQEhLNNcUl80iAoZNHaiXVIfjgfC7aPjUa3ZwHOgQWmhLRFgpNynQuSYXcjEYCDJ0AhkUcfNauXSv6+vpa21BG99jnQLkAlNvALjJ0AlK6Lq\/r2rVr4vr16\/IpaKm9yMZPmALhSFp24AY0MALrKkMnMIWhu4DKqlWrxPfffy9u3rzZgk7So+i2qmSwBKj8GnSZoROoErGXLTbS\/uOPP+Rm8T\/++KN8kqFDh8r\/B1CwB24TEhIDVWFju83QCVT1iGTGBtp0Ib0Clg8ul5tqByou7rZHEmDoeKQM7gpLoAkSYOg0Qcv8jCwBjyTA0PFIGdi7BMvg586dk71avHixwGFofLEE6iQBho5H2sQObbgAGgLQzJkzxerVqz3qJXeFJVBMAgydYvKzVlu3RwkyfE+ePCkOHTokI5LzXmxB5ZUc13MhAYaOC6nmuGfcFgLr168XBw4cEFOnTs1x1ztV2ILKLTqu6EACDB0HQs1zS51VU3SzJPTDpQWV5zm5DkuAoePJGHAFHZcWlCei424EJgGGjicKcwUHVzDzRGzcjQAlwNDxRGmupkEMHU8UzN1oSYCh49FgcOHwdWVBeSQ27kpgEigVOrR0i+TE9vZ27YoMXjy8KHHfk3xR7pFHHskVw2LDQetCzy6Wtl1ZUC6en+\/ZDAmUCh2IlPZbRbJiNP4EsEEgHM5Onjt3bqwGbMSv2LhHKEPEhQUVyrNzP\/2TQOnQgQgILngZKNqW\/srj+6RgOILWli1bEsGUJmpq77XXXit0n7R2fPjehQXlw3NxH8KUQCXQgagAnG+++UacOnVKjBkzRlo3+EmbVkV9FDR9wKZW77\/\/vujt7ZWawP1nzJghVq5cKT974IEHBtwb7V24cKFwxG+YqudeswSqkUBl0CGL5amnnhLLli2TcIDVk5ZnpE4VIDJAB3VxQB0BDGU6OjrEpEmT+n2mQo7q2oj4rUZ13KoqAbKedUmyZOlhDyIaI3HSK+IrbJL1XGT0VQYddZqFfz\/77LOpFkf04C8VOiqwdL4hfIYBpVpStqZqRRTAde1JAJYrkmWjPsEsVnTRXDf8Edy+fbs4cuSItOD5GiiBSqFjspqldpmsGgwscjTHfRa1YHTQ4Uzuer0SOouGxkeaFW3zD1ARa6leGtE\/TaXQQZfwV+jTTz9NNXtVq8Y2dMaNG8f71tRktBNk8EfprbfeEkuXLhWYwqftSxT1FRKE4Cs8duyY6OrqkhLC9I3uS59FLStdbFRNxGvlMRg6a9cK3rPGyljy5iY0vUaHVL9eUgejvkKCzo0bN1pTcpq+qYsSuqmbTavJG6Fa7EhQ0Eny6UStH5PpFQ8OiyPJs1vRYkJazBe6rRtX6kIHWUm6qZpues\/T9uTBEBR08Chxq1d5oKOL1vXs3eHu5JAAgQChEiZbvurAkQQiNUZMV1c3TnM8Rm2rBAcdXZwOlszzQIfjdOo3rlVnMvwxW7duTY1wdwWdq1evpq7I1k8D6U9UOXTSu9i\/hK0pEcdUZJV8GOWjy+bRIFTdU7iCDtpKc2CHIVW7vQwOOnh8G3lTNu5hVxV8t6IS0PlcdL6ZaDtJU6klS5a0AlZ1f\/DYp5Nda0FCh+bMdcsyz64+rkESMEkkVnP9opKLW73KAx1b1nhdtRssdOqqEH6ufBJYvny5PM89acsUpMbErWbFxenkgQ7H6STrkKGTb4xzrZpJwKZ1whHJDJ2avR78OK4kYMPPx7lX6dphSyddRlyiQRIoYqXwiqjZQPl\/aAqFwlGMs+QAAAAASUVORK5CYII=","height":137,"width":228}}
%---
%[output:1b5a5db1]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAR0AAACsCAYAAABGi3gfAAAAAXNSR0IArs4c6QAAIABJREFUeF7tXX+sVdWV3ibMBA2ivLT8EFSe2qaOo8IoBfIktSmJsdEWpWAfGccwNpqUqRhao8wwKsYUjZUBFQ02lhgir4WxJdX5w5SmTpAA0g4VYmxSBVToK9Y8tRJhpmSYfPvOuq633t5nn3PuOfuce+86ycu795599l7rW2t\/e6299znntFOnTp0yeigCioAiEAmB05R0IiGtzSgCioBFQElHHUERUASiIqCkExVubUwRUASUdNQHFAFFICoCSjpR4dbGFAFFQElHfUARUASiIqCkExVubUwRUASUdNQHFAFFICoCSjpR4dbGFAFFQElHfUARUASiIqCkExVubUwRUASUdNQHFAFFICoCSjpR4dbGFAFFQElHfUARUASiIqCkExVubUwRUASUdNQHFAFFICoCpZPOPffcYzZv3jxMqaefftrMnTs3qChdm7Y8KsxzTUiQUJ3btm0zt912m7eanp4es2HDBnPppZeGmqr0vNRjzpw5Zt26dWbMmDHmvffeM\/39\/Va+gYEBM378+GiyhvCHIPv37zeLFy+2dsAf9zuuB8rCnx566CErf29vbyp96BrUm2TrNKDIPlFEnWnaTVsmja0l3mnrRrnSSIcEP3jwoFOehQsXNg3vEziNs8lr81wTAixUZ91JB\/LhjzqaS1+fDtRhP\/nkk1qTDmy0Z8+eJoHwji1Jn5+LTTquQZgGy1bJLOTHac+nIR2SmWOetv7SSIfAlUblzp0lgkmrUBnlQqTD20xrsDLkdNVJI3SI5ElHGnVpJEOdVUdpIfxJVkTPRKx0zeWXX25ee+01G93gvBwMY5IOtf3RRx81MSX7yGgsln+42knrw9SXs0ZqpZCOC1yunOwI9B3h8csvv2wQHUGRAwcO2NSMkxMPjVEGBxyNFJcOSt\/XrFljnn\/+ebN9+3Z7De+Ex44dM0uWLGmew3nuBCGnT0M6coTjzu7THyNfSF+0zcvQd3Qw+XtSmkfy+Zzf54hJ8s2bN68ZHS1fvtzaaGhoyMLFbUqkQefkyB\/C35X60DWwM3yI7E0dBXoePnzYykLpYhob8Q4mCSxEHLy8r6MSFpdccomZNm2aefzxx62Msu5Q26HzhDGf+pDEjDJr1641S5cutX1S+g+XldLwNGRZCulww7qEIWHPOussa\/CtW7eOCP0BAOrhpCM7ERSkkSxEOr4oAJ0zFPKGnD5EOi65OfG5zuM3kK5MiaS+vrpd1yeRjkyvZGTkIp2QPYh0XCk2kS5wwFyRLMNlDeHvOk+\/ff\/73zc\/\/OEPzZQpU+z81KZNmyym9DuRjssHXTYiP\/NNH4SIR\/qajPZdBEz+JVNdiVna85ij8\/k85LnsssusTRCRoY\/ydvhgmTYikn2vEtKRwpLBZajLnWnWrFnNaMRl+BDpkEEAAEU1vtFGRmIhpw+RjgRdkjJ1BK4\/j758+lKn5uG6HH2o7lB65YqYeMQhbXbGGWcE7cFJh3QIpW1cb+qQSfhT+ddff31YGigjXJx\/4oknzFNPPWXwGcSzatWqYZEOt5PPRqSHK21N4yeuqJoTLCcd0l\/+RoMRtylvO3R+woQJdtIdB6XOpC\/qXLZsWXMgkL7Hfc2HfSjaqYR0fJGO7BgcSGJfHg7zjhIiHU4wvpUIOXKTPGmciYBOYn\/f6pCLGELpDGSaPXu2dR6elpAcRGBE6GlIB9e6OgUf\/Qh\/ilCS7OEiRZ+j+trlkaiMCtC2L5V3dUIe9dx\/\/\/3mW9\/61gjSCdlI+pmrg\/kGM15WRjRkr6NHj1qbIr3iWQLpg\/807eBrO3T+ggsusCtwPp9wYeqym2uACBEOzpdCOnnndKSxiiQd17yQj6iKjnQkyYVGUd6Zkjp1GaRDTiMjLT4\/g5Q4C+lwHaTzYtTlS\/E8gkoT6aQhHbTPV4b4aE6yETmTT4Rs5EtPeHSYpgPKyA\/X1IV0kuyG7R+1Ih0CHvMxch7BtXrlizyKTK98pEMdl+aXsP+E2i0i0qHOmbRiQZEOJ9286ZV0dB42+5bMQx0XcknSyZJeJTnvzp07bapDWPO5kjSkE0qveJRGcxMycnvsscfMHXfcYecwKN0gn6S0XNrIlV6FSMY11ymxJ9JB9CondiE\/n6vzRSoh2VwpLpdt5cqV5tZbbx0WBSZFOjK1DeFQSqTDR+o0+3TSkI5rNQbtpJ1I9pFO0mRnEaTDO6c0hs+hqVxoolaubvH65WoNzqVZvZIy0jUyIgE5h+STRIVrpPMinfDtT0lDOnyA4zb2DVgufZ555hlz3333DVu9JBx8NvJNJCdhnLR3jdpBf\/GlzKGJYqlbmsl5aW\/XAOOyGyKdWk0kc0VcYajMzdOSDurljo66caRZMk9Kr3j0hdyalglphF69evWIpXsfm7sMwfN3OAaf0MTISqO9ay4gpK\/EBN\/5CCgd3TUv4iM53oFCc0zU+bk90pAOnJf7CGTHAMPnHEJzai7\/kdfIyEVudqS5FEQYaW0ksU2z8zy0PYMvBMyfP9\/ceeed1jyhJXPZdhrZZN+Uk8ah9KpWS+ah8IoclCa00pTnxqJO5UpB0tTVDmXqrm+d5HNtDmwHG7tkzNuRq9C3VpsDQwDwmfgsW79d4XwobQjJUufzdde3TvLBp\/Jsya+b\/duJdPJiXtqcjsuYFPKNHTvWnr722msz3zwnQ8I0IW3dHCuLPHXXty7ytXIDYhZ7lF22XUinFbyjkw6IZ+rUqXZjWV9fX2bSKdvoWr8ioAiUi0BU0iFVaD7ARzq4J4buiylXfa1dEVAEcIsI\/mIdtSMdkM1dd91ldu\/eHQsDbUcR6GoEZs6caR555JFoxFM70tm1a5dZtGiRBWHy5Mld7QztpDwGCWw1ULu1k9WMHdxhN2x+xP2NMY7akk5MEGIA3eltHDp0yDz77LPmlltusXN2erQHAjTIx+xvSjrt4Ru1l\/LEiRNmcHDQTJo0yYwePboUeQf2\/NHW2z9jYin1d2OlSjrGmCpAKMrZ3hk6Yc7rKafD5ZURMu1460PTd+HZpcpWNulAj2kP7rR63H1Nr7nqorPzQqLXMQSq6G\/RIx25rV9uDqwChKxeiA4wsGfQ9F04bpjzo1OcO260WfTFSS2Nxq2Q1ytvfmh2vPWB7Zg4rl+315IOyPC3K2ZnVTV1+bJJ5+GXDpqHXzpk5VnXf7HFF79BX3yvguzJD\/C\/f8ak2hPhkoE3DGR9Ycn0pl2r6G9RSQcbinAvCR4dioM+87ckVAFC6p7FOjGuufuaqSM6N9WFETlvZ+hZ9itbDa8\/rYxwLKQh1DGJdHA9SEd2TpQtIgpykQ4cHMSA46qLxrVExKQXx4V0AwFBXzqgE9r9+benl0pGnAjz2CqtTYsox\/1gaPWXu4d0EOXs2LGj+XAi7GaV91\/VmXQwqn7tyb1Ng\/HogYiCOwg6A8qc13N66g4n2+AOksb5JGEh+kLn5xEC1cM7ctZ2pCySdNAm6keURQdIIG9axPGlDs5\/A+m8M3TcRj440G6ZRODSz4UhyuEvr95pbB4qI32KkzT628I7VppHV95tFlx7daiqQs5HjXT4XeGQXn7Hb3UmHYyg6Ej8gKPRfIO0CAiHd3ikXtL5KBpAOoTysg10JoyocBRKmSisR1n8ztM86ogUvdDkK8lGRMlHaTrXCvFI0pGOziOUrJ4r8QWZ4KB0C5851lJXsgGlra2mYpTC8vZdpE52kmkhypLdQY6UkuN\/Y5AaOS\/YSkTqszVkwL44DKQ3felvh0WLWW2UpXx00uGRDX8gEQkN0rnxgR+b+fO\/Yb74N+ebUaP+yo5gdJwzdpT9+Ic\/n7QRBM7hPx30nf\/Or\/\/UIU83J0\/+xdbTcNpGXXQsmP5Z6xi73\/64ef6VNz+wpMAPdAB0eh4BJRmAOgf+\/\/S2S82VD+2xxUESON79oDEy8o6T9J3KQY6Z559pbnx6fxb7O3VpyPApri78+IXADli+\/vb75vPnnG2u\/sJnzdItvx8W5XB9oCvmQNAOP9AOsISNt+z90zC7hAjGpzRIG7It3fJmswjaX7vgc2b328dGtM\/9AHKcO+70pg+Qn6ATc5tw3fjA0iCdT\/3FZ2MuBMpgYAIOV04ZbX62\/0NrC05c+A7ZXL4LH8AxatSoJr4ueeEv0JUGUZmi5naiFBfWknS+urnR+QnYKyaPNr85csJ+n3TmKDP48UkLOCcgKk86k0GoDMeCruV18fbwGW3iQFs4UBYyFHlcf\/EY88Ibwx1f1s+dK6ltiUUeOVEH9AY+UleSg+PpswHZi2xAstC1wJKwpTK4hrC+7gtjzIu\/O9a0NcrKutLqR\/Xy60lPrqM8z3XLYn\/ym7y+AtmgP\/nj06820kXpsy4M+bVoP0kG6VcdTToAkL8QjX\/HZ74j+YorrjATJ9ZnT8ba\/xwcFtJDXoykM88f04xYfJ2BynFj4zPqRLi+8O8+Yza9+scRo+\/aBRfZURqREdIrtIUogOYKsFKG0RV1nTx50sz6QWPOiUZMHiHQSIuR9MB7x8yWve8PExeR16ypDYfPehw\/ftzgQViwF\/bpICpc+KPfDasG7W+59dMJ3yxtnPPPO5vFUQ+fK8IJX3oFnYDNgmfesCM\/rv3uV85tDlhZZKCyN6zfbwlSRqDf\/coUgwiZExhsBfuibXRsRCLAeNehY+bRX75rq4QNEb0gKqI6uFyoD5EjdHz0hsaqpOuQREllcK209a7vTTfvfYIdybvMqu0fm9mXXjRsVSsPLmmviRrpyHSq3SeS4cC0\/OiaSMZ5SiHSrqTw+Quaf8myKU5OJPuWzPncEdrhc0ZpnYeXK3NOB+0kTSQTIVOqQARU1kQykcimVweb5Ff2loQ8NqFr+IIBDUjkt1XMoUYlnU5YMuerQTwk5UuSMCzO5d27QR0sT8hLctCSOTmcq1OA4DAXhUiJ793I4+CuJXNOFK12So4v36cDHeRyOc29Ubk8+qS5hk+Wl0VwaeQIlZGkw7dOdDzpAJx23xzInZ87Go9QeAQUcgjXeTgzNh\/SilaWOijt4qsgqM+3KpKl7qSyvs2BtNIjN1JmbZevwISW3qlsqFxWGXzlY+Dbqqx8VbOrNgdy4FypFc5XwbxZDErkQukIX95EFISjzJ2\/WWSNWTbmjuTQ0r7clR0Th7q2hcESqT4GMn7vWhX9LWp6RQZJekZyFSAU5ShKOuXd8EmpTKtRZFG2brd6fJsUq+hvUUknzTOSqwCh3RyojvKWHelA5ywT6nXEqI4yVdHfopNO6BnJVYBQR2doN5n0eTrtZrGGvFX0t6ikQ2ZJekYygbB06VKDxyjq0R4IHDlyxD5mVu3WHvYiKcluXf0QL31Gcns5rUrb\/gh0zDOS+at0YRb+Olt9G0T7O6pq0DkIdP3bIDrHlKqJIqAIuBCo3ZyOmkkRUAQ6GwElnc62r2qnCNQOgUpIx4cCfy82nwOqHWoq0DAEaP\/VwYONR5PigC3l868Vtvog4LojgNuxt7fXDAwMmPHjxxcudG1IZ9u2bWbVqlVW0X379jU\/l6F04Sh2eYVYNLj33nvN+vXrS3HSLoe3cPV9dwTwJ3m6nupZlCC1IR2uJK1u3XzzzWbu3LlF6ar1lIQABoyNGzc2n31dUjNabYsIJN0RQOeWL19u+1yZA0ktSEcuoYeW1FvEXi8vGAH5wP2Cq9fqCkIAxOK7I4A\/dgZvZ5HfCxLBVlMr0uGRje8u9CKV17qKQYDPxaHGOXPmaNRTDLSl1OIa1GVkA3K6\/fbbzQMPPGD4K6KKEEhJpwgUu7gOcmC8ThiPoZXfuxia2qqupGNM01H7+vrsioemV7X111SC8UUBXQhIBVnUQj7S4S+\/7Pj0CojzdEonkqP6YOGN6cRy4ZAWWqGLdGQ61fETyUBUl8wL9atolclVD\/p+00036T6daFbI1pAvk+i6JXOKdjZv3mwR1M2B2RypytJyc+DChQubrxmqUi5t242Aj3S6bnOgOogioAh0BwK1WL3qDqhVS0VAEQACSjrqB4qAIhAVASWdqHBrY4qAIqCkoz6gCCgCURFQ0okKtzamCCgCSjrqA4qAIhAVASWdqHBrY4qAIqCkoz6gCCgCURFQ0okKtzamCCgCSjrqA4qAIhAVASWdqHBrY4qAIqCkoz6gCCgCURFQ0okKtzamCCgCSjrqA4qAIhAVASWdqHBrY4qAIqCkoz6gCCgCURFQ0okKtzamCCgCSjrqA4qAIhAVASWdqHBrY4qAInDaL37xi1Oh94XTA5sB18DAgEnzLqM816QxB71N0vfgdvyOl75V+XBw\/sbLKuVIg2feMnhFyeLFi83Q0JCtore3d5hvhOyUt93QdWnaJd+cMWOG9RXyGVfdUq9Q+0WeJ4wvueSSlt+YKt\/Ciu94x1wVx2m9vb2nQh0jD4HkuSYNACGnyks6MPD69eutE44ZMyaNKM4yeJUON2YI29wNVXihJBwShXfQkJ3KEj9Nu\/AR\/G3YsMG+MrfTSUcSDmFfFfFY0oEQ7fLKl5BT5SEdIooi3sFNdXUi2ZCzSoz5q0uq9qOQf5CsU6ZMaUYPpE9VndBHwEVEOqTvRx99NIJki\/D3PIPHaevXrz\/F0xEyAELnl19+2Rw8eNB85zvfMS+++KKtn6dXfISQL+qaN2+e6e\/vH3YNOcSaNWvM888\/b7Zv327P8w5K7+ShczjPwQk5lewQ3KEOHDhg6L1aVOeuXbtGhJnUceQIyDuUC6err77aYkZHT0+PNTQigCVLljT19aUiwPm5556zKQvqnzVrlvc6HkkuX77cviGVUh3Z8eVIJ88n6elyKiqflHq47JTkL4gOY\/iHa1BISzpJOJHsvN+Q\/YEhpaKyo0vbcExdpCPfMRYiDl7eR6q8nWnTppnHH3\/cml3WHWo7dB51QobT9u3bdwqAnHXWWZZQtm7dOuJFaY888oh58sknhxGIKyS9\/PLLzWuvvWYrTiIdnyNjbikUCuYlHVeb\/FXG\/Dx0A0GBjOXhIySUmz9\/viVTTjrAbd26dU3CSUpFslyHsiB1DArycKU5WfQIRb4yvXI5vrRTyF846cTwD94Bk9KrJHtznHx+6\/M76OtrlwZhSTqffPKJ0+Yh4gkNOr50mRNPqO3QeUxZkBynHT169BSPSIh0uOPK+ZkzzjijOQKT8SSjJpEOgQSlKALwsbCMXPKSjqsjknFlekUA8pBUOsCmTZssKcnRXo6krtRNjqykE3ee0HWEL0iHsCMZgSsiLBph6TvmL7h8y5Yts06cpKdvfkvOXaENHrFyO\/GIzecvnHTK9I9QBOYi58suuyyI04MPPmijaJIddqHohnQOpf7S5lQHTSSTz\/lw9i0IubIHisLgE5x0iGTlbzQI+9oOnZ8wYYLFA4c30uGVS9KhURb\/XelWKNJxjTKSdOQoQPLkJR2uT4gYuMP4oggiZzl3I+tOGkWlTmlHX1znIgxyrtdff92SztGjR23q6JtfShrh0q7ayFGUdOB2ok6b5C+cdMryD975XKlyKP2g9JX7BOG0evVqSzqSVDmh++b7JIFL4iLSIWJLip5c5+g334oj\/ASEIFfJyIb4z6cmZBtpzl9wwQVNX\/TO6XADFE06SQb3heWtkg7Xp0jSkY7aCum4cHE5EScd3pGrIB2Sj+zmslNW0inLP6ogHW4fn2+QD4UinVZJR5KPjIijkY5cvXJNqhWdXvmcavbs2ZZxaX4J+4GSnNkVTsoQ1qWPNL4vd+ajlCvsRnoVIp00K2Ou6C10nWtLgiSdpPQKo+nKlSvNrbfeOiy9ShopcS7UcV2kkzW9KtM\/ktIrX6TjWgFyjfauSMdHOq5IlXzVF+m40quQvVx+JPUhP6EFDPQruSLpSp9426HUkaf+I\/bppCEdkEFoYjBpTsfnVHyeQoJZZqQjw86kiWSSw7fiIQnNlU+TboSDqyOErnNFD5J0kK\/L9IdHJmkmM11OnWbi1Rex8vr4wgNPr2L4R5oUztWpQn4p0ysf6axYsWLYyiSv10c6vslaPj8j5ZMrSkntuNJHkiXUNuZsXAsbXDbyiRE7ktOSDoTnzocKcdDon4d04Hg8v0W+vHbtWrN06VJbN+aPKHfmjpnEuGkiHdnByXF880pc91CkI6MDSTj47punchEP6Z0m0gHp8Pol4XBZ+Epdmj1GrolkbpOkiIJk4v6ShnSK8I8il8xdk6ppSQd488EOnfOJJ54wTz31lKE5OYpAeNojSSSJcMi+oW0oPNLHCuydd95pLw0tmcu208gGfHLde8WVIOD5b74w1TVq6m+dj0Cd\/MW1ObDzLZCsYRGbELNgmIt0ZJTDG0zDvDTS0Yx2FoG1bHsi4EvJ0vpLkVpDFvzRbRBF1t2OdbUN6bjC9rQOxJfiqrrprB2do91llvNLaf2laL3lDZ9F199u9bUV6WQFl4w9duxYe+m1115b2Z2uWWXX8oqAIlAMArnTqzzNg3TwN3XqVDtz39fX5ySdw4cPG\/zpoQgoAuUjgJtf8RfriEo6pBRNLLpIB2Rz1113md27d8fCQNtRBLoagZkzZxrcXxmLeGpHOrjre9GiRRaEyZMnd7UztJPyGCSwvaFKu737wQlz7rjR7QRb5bKS3bDxEJs4Yxy1JZ2YIGQFesnAG2bHWx+a366YnfXSji1\/6NAh8+yzz5pbbrnFps+xj3eGTphpD+40\/TMmmnX9F8duvm3bo0E+Zn9T0snhLj3LfmWvGlr95RxXd+YlJ06cMIODg2bSpElm9Oj40QaRznk9o3UwyOBiSjrGmCpAyGAjQ86NaxDpwMljH9ev22sWfXGSHdXrcnQC6cC2+LvqorPrAmvpclTR3zTS8ZgVKRQOGapz0vn5t6dX4qCItO6+Zqq5+5re0p0ybQN1IZ1WBoOBPX80sHs3RbBdQTryfi25ObAKEFwdC9EEJiblvE1ZpAOHxyQoRlk4fv+MSV5CA+n0XXi2eWHJ9LScUHq5qknnlTc\/NF97cm9zoMgTBT780kHz8EuHlHRK9paokQ52PuJmMjwjGQd9phsT8VudSAeTxTKa4c5dZLQBkkOqhsgqKZKh0biVEd3lU9ALnQ7t50kZqyQdjgl0yzuZTHN1wCAPaZXcVwutHvbe8dYH5oL\/fdvcveQfTcdOJCPK2bFjR\/Mp\/NgWL++\/ykI6AA7RCEZ92VFwDr\/l6UCwLlZCENXIzs1JJ69zJ0VWSJkotUOUBeKjCAjX0WicRDqQcWDPoCUQ0iGEA9ULLIHpE9+8OHXqCMK8cspo8w+XjbYTyb8+fKIl7LP2Lkk6pGsWHdAmkU6Rds2qS9Hlgc07Q8dHpOKk6829H5kX\/m1Z55IOf\/wFwJXfeaSDx1l8\/etfNxMnDp8s\/cOfT1q7bNn7JxsK44CTobM8ekOvwXkw+NItb9rff3rbpeacsaOG2RJl+G\/yOwrP+sHeYR0WnRdt4AAR8GPB9M+YtQs+Z9umel110jWu9q98aM8If4P81C4mjm+49Gyz4JnGcj0OkAp+4\/UCl4ajnbDzPvR51\/emm1GjRg2TcemW31uMvvuVcw0+c73Q8ZDigUwgL\/5ARnDgBdM\/29STy7P17yea\/\/nrs83CH\/3OioT2z+s5fZiMuw4ds226cKJ20Kb1hUPHzKyp\/neQofzut4+ZV978wOrpOiADjpnnnzmsXfxGskMeTvaELf5LfKkNyI\/2gfOd\/\/57Gxkt\/VKDcHHgPJUhXX+2\/0Nz8uRfLH44qCz0xbmGnA19oRe1n+RLUmeUffSX7zZ1RT8gfTCwYACDr5K\/wad\/8x\/Pmp\/e+83O3KcjIxvXw7IQ6Xx183GnA4V+vGLyaDP4caODyIM7Oc7dP\/cz5jdHThhc88Ibx5rXJdURah9t4HrU65OBfieHJAd1lQ+1R9dOOvNTUkXbaQ6SM01ZLmseeQl7yMntc\/3FY5o4SbvhGiqP\/7wO6Ig\/KVdIF65zWv354AQ5OL4kI\/0mbcrl4dfK65LkJp+Cj3LCu+4LY5r6E3bkzy4fCGEVM7qLOqeTlnS+cc86++Au37ZsjLYU5XCD\/fqeGfYrGH3L3vftZ8rNr7ponGV5jNY4aESBMTDi0O\/k6KiDUpOQM9NIiWsRTZw8edJ2LnlQ3a7fb3x6f6gZqwuiES4XtY2LfbgQDriO9ERdwBD\/gQ2ldCQE6qW0Dvo8+svDNtKjdBbXENaolzoS9ug0IqIGzo1IakqTXBD5IDKhc1xpHrlCrr4Lx1ksZWSEaxAJbHp1cETUyeujSJHLsfm\/3rfyIbKBjXHQLmYZwRJujTT99GH4Eu64FhPYFG0TxpTyU5qPuqh+XEOpP8qTnIjWqB6UBQZoF75KkQ\/pB19CFIhokA74HCIpim64\/pCH2uef6dqY81jRSQdK0lPqktKr0MSWnNBFKkC5PE2Koq28KzyYp+BGos4Kg8lQvoilcz6HRI7QcLpGqkH6gRyofTiKnM+CE6MM7TUBFvjsWl4nh0f9sv20Ix\/ZAZHD0zdObM7pEDll3ffCV\/GCLPz\/AwylkLI8ESoRS9L+G74qmbTBkPby8PlCjmOSzGnLNQaQBhnlOcg\/QG7\/9OM3LKkiPW+sijZ2bPM5nZ9tXG82P3ZfZ6ZXMp1qdSIZIxVGgjJWGoh0ZOfjk5bo8EVt0qNbK4jUfJ2e2i96yZzfRkAEl9bpIdPx48fNl881lexIdk0k05xUWh3QucnmRa5K5iGNsq7hREa6\/ssV\/23W\/eu3O3ciud2WzF37dOAQZa9ypFkyTxuJZHHgRoQ0LheJV7lkTsu\/lJ7lTRWIvIqIXLPgXkVZEBDSQqxedTTpANx22hzoS8\/K3s8RWu5vhRzKcvAqSYd0ok2TefcatZrWlIVtmfVm2aJSlBxR53S40K7UCuerAMEFJlI3dH7XnBCcG2E7RsQs4XtRRqtjPXUinbzzeHXEtWyZquhvlZBO0jOSqwAhq2HreBtCVh2KLq+kUzSiceqror9FJZ00z0iuAoSs5qWl1jrdcJnm0Mk3AAAE+UlEQVRVh6LLV\/08HZprK2Ouq2is6lRfFf0tOumEnpFMIGCfDh6jqEd7IHDkyBH7mNkq7YaJZKS7ZaxmtocVsktJdgttUcles\/+KqKRDYugzkos0odalCLSGQNc\/Ixnw6dsgWnMivVoRyIJAx7wNgr+nGQDw91wnRTpZwNKyioAi0H4I1C69aj8IVWJFQBHIgoCSTha0tKwioAi0jECtSIe\/65qnYy1rqRWUigBthTh4sHHXNg7YUt9TXyrsLVXu2pzL7djb22sGBgbM+PHjW2rHdXElpOMSZNu2bWbVqlVW0X379jU\/l6F04Sh2eYWYv7v33nvN+vXrS3HSLoe3cPV9m3P5Ux9cT4AoSpDakA5Xkiaab775ZjN37tyidNV6SkIAA8bGjRubj6EtqRmttkUEkjbn0rnly5fbPlfmQFIL0pGrWbq61aJ3Rb5cPvs6cvPaXEoEQCy+zbn8CRB4UYL8nrKJVMVqRTo8svHdEJpKKy0UFQE+F4eG58yZo1FPVAtka8w1qMvIBuR0++23mwceeMDwt7Vka8ldWkmnCBS7uA5yYLwFAk+ElN+7GJraqq6kY0zTUfv6+uyKh6ZXtfXXVILxRQFdCEgFWdRCPtLh76Hr+PQKiPN0SieSo\/pg4Y3pxHLhkBZaoYt0ZDrV8RPJQFSXzAv1q2iVyVUP+n7TTTfpPp1oVsjWkC+T6Lolc4p2Nm\/ebBHUzYHZHKnK0nJz4MKFC5tv\/KhSLm3bjYCPdLpuc6A6iCKgCHQHArVYveoOqFVLRUARAAJKOuoHioAiEBUBJZ2ocGtjioAioKSjPqAIKAJREVDSiQp3NY3Jpzi6pFixYoV57rnnzJo1awrf9p6kddo9PWXuG6nGKt3bqpJOF9q+zMcWZIEz6\/092EZx4MABXY7PAnINyyrp1NAoZYtUF9LJSiJZSapsHLX+fAgo6eTDra2vcpEOv9dmwoQJpr+\/31x33XU25RoaGjI9PT1mw4YNZufOnc1IQ95NjlSJPy0w6emBciczAKVNa9u3b7f4Upv8LuesRNXWhupQ4ZV0OtSwSWqlJR3UQY+slE+bk7c7gAx+8pOfNMuHbodw3RQq5XI9p0dvJm1\/h1XSaX8bZtYgLenw+6dkZ+db6RctWmSWLFli5JMekwhCEkraR2KUefdzZiD1glwIKOnkgq29L0pLOvToSmibRDrz5s2z6Rh\/MDsh5HvAty+KofTMlVqhTlda1t7W6D7plXS6z+b2MSI48NAtOlxzOmlJZ\/bs2Wbx4sW2vrTPtE56xCmf25Hko6TT\/g6rpNP+NsysQdGkQ5FOlsdZpHmucpqHTWVWXi+oHAElncpNEF+AokkHKRFIBH9Y4aLVJrQzODjofF6y3OznmtNxzd\/oRHJ8fym6RSWdohFtg\/rKIB2oDdLhKVvSA9pdT4d0vbRPPlepLnuM2sDMtRVRSae2pul8wbLuudHNgZ3hE0o6nWHHttQiK4lkJam2BKULhFbS6QIj11lFveGzztYpRzYlnXJw1VoVAUXAg4CSjrqGIqAIREVASScq3NqYIqAIKOmoDygCikBUBJR0osKtjSkCioCSjvqAIqAIREVASScq3NqYIqAIKOmoDygCikBUBJR0osKtjSkCioCSjvqAIqAIREVASScq3NqYIqAIKOmoDygCikBUBJR0osKtjSkCioCSjvqAIqAIREXg\/wAF9dfqmrt8KgAAAABJRU5ErkJggg==","height":137,"width":228}}
%---
%[output:5601ebfe]
%   data: {"dataType":"textualVariable","outputData":{"name":"preRIR_sirCheck_0dB","value":"1.9287e-15"}}
%---
%[output:5a4f8c86]
%   data: {"dataType":"matrix","outputData":{"columns":2,"name":"anechoic_postRIR_SirCheck_0dB","rows":1,"type":"double","value":[["0","0"]]}}
%---
%[output:6197802b]
%   data: {"dataType":"matrix","outputData":{"columns":1,"name":"anechoic_snrCheck_5dB","rows":2,"type":"double","value":[["5.0001"],["5.0018"]]}}
%---
