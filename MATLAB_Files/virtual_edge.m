% Initialize the node2_pipe structure
node2_pipe = struct('point_idx', [], 'n1_idx', [], 'n2_idx', [], 'cos_angle', [], 'shortest_dist', []);

% Iterate through each link in link2
for i = 1:length(link2)
    % Get the middle point of the link
    point_indices = mlink2(i).point; 
    num_points = length(point_indices);
    mid_point_idx = point_indices(ceil(num_points / 2));
    
    % Get the start and end point indices
    n1_idx = link2(i).n1;
    n2_idx = link2(i).n2;
    
    % Calculate direction vectors
    [mid_point_x, mid_point_y, mid_point_z] = ind2sub([w,l,h], mid_point_idx);
    mid_point = [mid_point_x, mid_point_y, mid_point_z];
    
    n1_point = [node2(n1_idx).comx, node2(n1_idx).comy, node2(n1_idx).comz];
    n2_point = [node2(n2_idx).comx, node2(n2_idx).comy, node2(n2_idx).comz];
    
    dir1 = mid_point - n1_point;
    dir2 = mid_point - n2_point;
    
    % Check if any direction vector is zero
    if any(dir1 == 0) || any(dir2 == 0)
        % Handle zero vector case
        cos_angle = -1.0000; % Set cosine value to -1
    else
        % Calculate cosine of the angle between the two direction vectors
        cos_angle = dot(dir1, dir2) / (norm(dir1) * norm(dir2));
    end
    
    % Check if cosine value is NaN
    if isnan(cos_angle)
        % Handle NaN case
        cos_angle = -1.0000; % Set cosine value to -1
    end
    
    % Calculate the shortest distance from mid_point to the line segment n1-n2
    line_vec = n2_point - n1_point;
    point_vec = mid_point - n1_point;
    line_len = norm(line_vec);
    line_unitvec = line_vec / line_len;
    point_vec_scaled = point_vec / line_len;
    t = dot(line_unitvec, point_vec_scaled);
    
    if t < 0.0
        t = 0.0;
    elseif t > 1.0
        t = 1.0;
    end
    
    nearest = line_vec * t;
    dist_vec = point_vec - nearest;
    shortest_dist = norm(dist_vec);
    
    % Append to node2_pipe structure
    node2_pipe(i).point_idx = mid_point_idx;
    node2_pipe(i).n1_idx = n1_idx;
    node2_pipe(i).n2_idx = n2_idx;
    node2_pipe(i).cos_angle = cos_angle;
    node2_pipe(i).shortest_dist = shortest_dist;
end

%% Initialize lung and lobe properties
for i = 1:length(node2_pipe)
    node2_pipe(i).lung = 0;
    node2_pipe(i).lobe = 0;
    node2_pipe(i).ep = 2; % 0: unknown, 1: artery, 2: vein
end

% Label lung and lobe based on coordinates
for i = 1:length(node2_pipe)
    % Get coordinates
    [x, y, z] = ind2sub([w, l, h], node2_pipe(i).point_idx);
    
    % Get label value from lobe map
    label_value = lobe(x, y, z);
    
    % Assign lung and lobe labels based on the label value
    if label_value == 1 || label_value == 2
        node2_pipe(i).lung = 1;  % Left lung
        node2_pipe(i).lobe = label_value;  % Lobe type
    elseif label_value == 3 || label_value == 4 || label_value == 5
        node2_pipe(i).lung = 2;  % Right lung
        node2_pipe(i).lobe = label_value;  % Lobe type
    else
        node2_pipe(i).lung = 0;  % Unknown
        node2_pipe(i).lobe = 0;  % Unknown
    end
end

%% Assign x, y, z coordinates to node2_pipe
for i = 1:length(node2_pipe)
    pipe_point = node2_pipe(i).point_idx;
    [x, y, z] = ind2sub([w, l, h], pipe_point);
    node2_pipe(i).comx = x;
    node2_pipe(i).comy = y;
    node2_pipe(i).comz = z;
end

%% Compute offsets for real and simplified points in the link
for i = 1:length(link2)
    % Get real and simplified points
    real_points = mlink2(i).point; % Real points in mlink2
    real_points_coords = zeros(length(real_points), 3);
    
    for j = 1:length(real_points)
        [x, y, z] = ind2sub([w, l, h], real_points(j));
        real_points_coords(j, :) = [x, y, z];
    end
    
    % Get the points of the start and end nodes
    n1_point = [node2(link2(i).n1).comx, node2(link2(i).n1).comy, node2(link2(i).n1).comz];
    n2_point = [node2(link2(i).n2).comx, node2(link2(i).n2).comy, node2(link2(i).n2).comz];
    line_vec = n2_point - n1_point;
    line_len = norm(line_vec);
    line_unitvec = line_vec / line_len;
    
    % Initialize normal and tangent offsets
    normal_offsets = zeros(length(real_points), 1);
    tangent_offsets = zeros(length(real_points), 1);
    
    % Compute the normal and tangent offsets for each real point
    for j = 1:length(real_points)
        point_vec = real_points_coords(j, :) - n1_point;
        t = dot(line_unitvec, point_vec / line_len);
        
        tangent_offsets(j) = t * line_len;
        
        % Calculate normal offset
        nearest_point = n1_point + t * line_vec;
        normal_offsets(j) = norm(real_points_coords(j, :) - nearest_point);
    end
    
    % Store average offsets
    node2_pipe(i).avg_normal_offset = mean(normal_offsets);
    node2_pipe(i).avg_tangent_offset = mean(tangent_offsets);
end

%% Compute curvature difference for each link
for i = 1:length(link2)
    % Get real and simplified points
    real_points = mlink2(i).point;
    simplified_points = link2(i).point;
    
    % Calculate curvatures
    real_curvature = calculate_curvature(real_points);
    simplified_curvature = calculate_curvature(simplified_points);
    
    % Calculate the difference in curvature
    curvature_diff = abs(real_curvature - simplified_curvature);
    
    % Store the curvature difference
    node2_pipe(i).curvature_diff = curvature_diff;
end

%% Curvature calculation function
function curvature = calculate_curvature(points)
    % points: an N x 3 matrix representing 3D coordinates (x, y, z)
    num_points = size(points, 1);
    
    % Initialize total curvature
    total_curvature = 0;
    
    % Calculate curvature between adjacent points
    for j = 2:num_points-1
        prev_point = points(j-1, :);
        curr_point = points(j, :);
        next_point = points(j+1, :);
        
        % Compute vectors
        vec1 = prev_point - curr_point;
        vec2 = next_point - curr_point;
        
        % Compute angle
        len1 = norm(vec1);
        len2 = norm(vec2);
        
        if len1 > 0 && len2 > 0
            cos_theta = dot(vec1, vec2) / (len1 * len2);
            theta = acos(min(max(cos_theta, -1), 1));
        else
            theta = 0; % Assign zero curvature for degenerate cases
        end
        
        total_curvature = total_curvature + theta;
    end
    
    % Compute average curvature
    if num_points > 2
        curvature = total_curvature / (num_points - 2);
    else
        curvature = 0; % Set curvature to 0 if not enough points
    end
end
