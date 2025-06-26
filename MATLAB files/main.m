%% Initialization
clear all;
close all;

%% File paths (replace with relative paths for GitHub)
% Define input and output file paths
input_paths = struct();
input_paths.skeleton = 'data/skel.nii.gz';          % Vessel skeleton
input_paths.reconstruction = 'data/data.nii.gz';    % Reconstructed vessel data
input_paths.lobe_labels = 'data/label_lobe.nii.gz'; % Lobe labels
input_paths.airway = 'data/label_airway.nii.gz';    % Airway labels
input_paths.dyeing = 'data/data_dyeing.nii.gz';     % Dyeing data for clustering

output_paths = struct();
output_paths.y_file = 'output/y.txt';               % Output node features
output_paths.x_file = 'output/x.txt';               % Output edge features
output_paths.edge_file = 'output/edge.txt';         % Output edge list
output_paths.len_file = 'output/len.txt';           % Output edge lengths
output_paths.mat_file = 'output/graph_data.mat';    % Output MAT file

%% Load and preprocess data
% Load NIfTI files
info = load_nii(input_paths.skeleton);
image = info.img;

recreate_info = load_nii(input_paths.reconstruction);
recreate = recreate_info.img;

lung_label_info = load_nii(input_paths.lobe_labels);
lung_labels = lung_label_info.img;

classified_info = load_nii(input_paths.reconstruction);
classified = classified_info.img;

airway_info = load_nii(input_paths.airway);
airway = airway_info.img;

dyeing_data_info = load_nii(input_paths.dyeing);
dyeing_data = dyeing_data_info.img;

%% Data resizing (if needed)
% Resize data if height dimension is less than 300
[~, ~, h] = size(image);
if h < 300
    image = resize_data(image);
    airway = resize_data(airway);
    lung_labels = resize_data(lung_labels);
    classified = resize_data(classified);
    recreate = resize_data(recreate);
    dyeing_data = resize_data(dyeing_data);
end

%% Graph structure extraction
% Extract graph structure from skeleton and airway
[mlink2, link2, node2] = get_struct(image);
[mtrachea_links, trachea_links, trachea_nodes] = get_struct(airway);

%% Feature engineering
% Add node features
add_node_features;

% Perform auxiliary task (clustering for feature enhancement)
auxiliary_task;

%% Save results
% Save graph data to text files for Python processing
node_link2numpy_txt;

% Save all data to MAT file
save(output_paths.mat_file);
