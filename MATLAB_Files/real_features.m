function [node2, link2] = real_features(node2, link2, mlink2, w, l, h, lobe, image, extra_PAV, trachea_nodes, trachea_links)
    % ADD_NODEFEATURES Enhances node and link structures with additional features
    % Inputs:
    %   node2, link2 - Graph structures to be enhanced
    %   mlink2 - Original detailed link structure
    %   w, l, h - Volume dimensions (width, length, height)
    %   lobe, image, extra_PAV - Volume data arrays
    %   trachea_nodes, trachea_links - Trachea graph structures
    
    % Add basic link length features
    [node2, link2] = add_link_length_features(node2, link2, mlink2);
    
    % Add projection distance features
    [node2, link2] = add_projection_features(node2, link2, w, l, h);
    
    % Add Manhattan distance features
    [node2, link2] = add_manhattan_features(node2, link2);
    
    % Add branch direction features
    [node2, link2] = add_branch_direction_features(node2, link2);
    
    % Add angle similarity features
    [node2, link2] = add_angle_similarity_features(node2, link2);
    
    % Add lobe and lung classification
    node2 = add_lobe_classification(node2, lobe, w, l, h);
    
    % Add artery/vein labels
    node2 = add_artery_vein_labels(node2, image, w, l, h);
    
    % Add extra pulmonary artery/vein markers
    node2 = add_extra_pav_markers(node2, extra_PAV, w, l, h);
    
    % Add radius features
    [node2, link2] = add_radius_features(node2, link2, image);
    
    % Add direction vectors and lengths
    link2 = add_direction_vectors(link2, node2);
    
    % Add curvature features
    node2 = add_curvature_features(node2, link2);
    
    % Add node degree features
    node2 = add_degree_features(node2);
    
    % Add resistance features
    link2 = add_resistance_features(link2);
    
    % Add trachea proximity features
    [node2, link2] = add_trachea_features(node2, link2, trachea_nodes, trachea_links, w, l, h);
    
    % Add geometric deviation features
    link2 = add_geometric_deviation_features(link2, node2, mlink2, w, l, h);
    
    % Add direction deviation features
    link2 = add_direction_deviation_features(link2, node2, mlink2, w, l, h);
    
    % Add topological deviation features
    link2 = add_topological_deviation_features(link2, node2, mlink2);
    
    % Add combined deviation features
    link2 = add_combined_deviation_features(link2);
end

%% Helper functions for each feature category

function [node2, link2] = add_link_length_features(node2, link2, mlink2)
    % Add link length features based on point counts
    for i = 1:length(link2)
        link2(i).link_length = length(mlink2(i).point);
    end
    
    for i = 1:length(node2)
        node2(i).link_length = zeros(size(node2(i).links));
        for j = 1:length(node2(i).links)
            link_idx = node2(i).links(j);
            node2(i).link_length(j) = link2(link_idx).link_length;
        end
    end
end

function [node2, link2] = add_projection_features(node2, link2, w, l, h)
    % Add projection distance features
    for i = 1:length(node2)
        node2(i).x_projection = zeros(size(node2(i).links));
        node2(i).y_projection = zeros(size(node2(i).links));
        node2(i).z_projection = zeros(size(node2(i).links));
        node2(i).link_length_abs = zeros(size(node2(i).links));
        
        for j = 1:length(node2(i).links)
            link_idx = node2(i).links(j);
            [start_point, end_point] = get_link_endpoints(link2(link_idx), node2, w, l, h);
            
            % Calculate projections and distances
            vec = end_point - start_point;
            link_length = norm(vec);
            
            node2(i).link_length_abs(j) = link_length;
            node2(i).x_projection(j) = abs(vec(1));
            node2(i).y_projection(j) = abs(vec(2));
            node2(i).z_projection(j) = abs(vec(3));
            link2(link_idx).link_length_abs = link_length;
        end
    end
end

function [start_point, end_point] = get_link_endpoints(link, node2, w, l, h)
    % Get start and end points of a link
    if length(node2(link.n1).idx) > 1
        start_point = [node2(link.n1).comx, node2(link.n1).comy, node2(link.n1).comz];
    else
        [x, y, z] = ind2sub([w,l,h], node2(link.n1).idx);
        start_point = [x, y, z];
    end
    
    if length(node2(link.n2).idx) > 1
        end_point = [node2(link.n2).comx, node2(link.n2).comy, node2(link.n2).comz];
    else
        [x, y, z] = ind2sub([w,l,h], node2(link.n2).idx);
        end_point = [x, y, z];
    end
end

function [node2, link2] = add_manhattan_features(node2, link2)
    % Add Manhattan distance features
    for i = 1:length(link2)
        start_point = [node2(link2(i).n1).comx, node2(link2(i).n1).comy, node2(link2(i).n1).comz];
        end_point = [node2(link2(i).n2).comx, node2(link2(i).n2).comy, node2(link2(i).n2).comz];
        link2(i).Manhattan_distance = sum(abs(end_point - start_point));
    end
    
    for i = 1:length(node2)
        node2(i).Manhattan_distance = zeros(size(node2(i).links));
        for j = 1:length(node2(i).links)
            link_idx = node2(i).links(j);
            node2(i).Manhattan_distance(j) = link2(link_idx).Manhattan_distance;
        end
    end
end

function [node2, link2] = add_branch_direction_features(node2, link2)
    % Add branch direction features
    for i = 1:length(node2)
        node2(i).branch_directions = zeros(length(node2(i).links), 3);
        for j = 1:length(node2(i).links)
            link_idx = node2(i).links(j);
            start_point = [node2(link2(link_idx).n1).comx, node2(link2(link_idx).n1).comy, node2(link2(link_idx).n1).comz];
            end_point = [node2(link2(link_idx).n2).comx, node2(link2(link_idx).n2).comy, node2(link2(link_idx).n2).comz];
            node2(i).branch_directions(j, :) = end_point - start_point;
            link2(link_idx).branch_direction = end_point - start_point;
        end
    end
end

function [node2, link2] = add_angle_similarity_features(node2, link2)
    % Add angle similarity features (cosine and sine)
    for i = 1:length(link2)
        % Find connected links
        prev_links = find_connected_links(link2, link2(i).n1, i);
        next_links = find_connected_links(link2, link2(i).n2, i);
        
        % Initialize arrays
        link2(i).cos_n1 = zeros(1, length(prev_links));
        link2(i).sin_n1 = zeros(1, length(prev_links));
        link2(i).cos_n2 = zeros(1, length(next_links));
        link2(i).sin_n2 = zeros(1, length(next_links));
        
        % Calculate similarities for previous links
        for j = 1:length(prev_links)
            [cos_sim, sin_sim] = calculate_angle_similarity(link2(i).branch_direction, link2(prev_links(j)).branch_direction);
            link2(i).cos_n1(j) = cos_sim;
            link2(i).sin_n1(j) = sin_sim;
        end
        
        % Calculate similarities for next links
        for k = 1:length(next_links)
            [cos_sim, sin_sim] = calculate_angle_similarity(link2(i).branch_direction, link2(next_links(k)).branch_direction);
            link2(i).cos_n2(k) = cos_sim;
            link2(i).sin_n2(k) = sin_sim;
        end
    end
end

function links = find_connected_links(link2, node_idx, exclude_idx)
    % Find links connected to a node (excluding specified link)
    links = find(([link2.n1] == node_idx | [link2.n2] == node_idx));
    links(links == exclude_idx) = [];
end

function [cos_sim, sin_sim] = calculate_angle_similarity(vec1, vec2)
    % Calculate cosine and sine similarity between vectors
    cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2));
    cross_product = cross(vec1, vec2);
    sin_sim = norm(cross_product) / (norm(vec1) * norm(vec2));
end

function node2 = add_lobe_classification(node2, lobe, w, l, h)
    % Classify nodes by lobe and lung
    for i = 1:length(node2)
        [x, y, z] = get_node_coordinates(node2(i), w, l, h);
        label_value = lobe(x, y, z);
        
        % Determine lobe and lung
        if label_value == 1 || label_value == 2
            node2(i).lung = 1;  % Left lung
            node2(i).lobe = label_value;  % 1: upper, 2: lower
        elseif label_value >= 3 && label_value <= 5
            node2(i).lung = 2;  % Right lung
            node2(i).lobe = label_value;  % 3: upper, 4: middle, 5: lower
        else
            node2(i).lung = 0;
            node2(i).lobe = 0;
        end
    end
end

function node2 = add_artery_vein_labels(node2, image, w, l, h)
    % Add artery/vein labels to nodes
    for i = 1:length(node2)
        node2(i).label_arteryvein = zeros(1, length(node2(i).idx));
        for j = 1:length(node2(i).idx)
            [x, y, z] = ind2sub([w, l, h], node2(i).idx(j));
            if image(x, y, z) == 2
                node2(i).label_arteryvein(j) = 2; % Artery
            elseif image(x, y, z) == 1
                node2(i).label_arteryvein(j) = 1; % Vein
            end
        end
    end
end

function node2 = add_extra_pav_markers(node2, extra_PAV, w, l, h)
    % Add extra pulmonary artery/vein markers
    for i = 1:length(node2)
        [x, y, z] = get_node_coordinates(node2(i), w, l, h);
        if extra_PAV(x,y,z) > 0
            node2(i).extra_arteryvein = extra_PAV(x,y,z);
        else
            node2(i).extra_arteryvein = -1;
        end
    end
    
    % Display statistics
    extra_values = [node2.extra_arteryvein];
    has_positive = any(extra_values > 0);
    if has_positive
        fprintf('Found extra-pulmonary nodes: %d type 1, %d type 2, total %d\n', ...
            sum(extra_values == 1), sum(extra_values == 2), length(extra_values));
    else
        disp('All nodes have extra_arteryvein <= 0');
    end
end

function [x, y, z] = get_node_coordinates(node, w, l, h)
    % Get node coordinates (either COM or from idx)
    if length(node.idx) > 1
        x = round(node.comx);
        y = round(node.comy);
        z = round(node.comz);
    else
        [x, y, z] = ind2sub([w, l, h], node.idx);
    end
end

function [node2, link2] = add_radius_features(node2, link2, image)
    % Add radius features to nodes and links
    for i = 1:length(node2)
        node2(i).r1 = zeros(1, length(node2(i).links));
        node2(i).r2 = zeros(1, length(node2(i).links));
        node2(i).r_node = cal_radius([node2(i).comx, node2(i).comy, node2(i).comz], image);
    end
    
    for i = 1:length(link2)
        n1_sub = [node2(link2(i).n1).comx, node2(link2(i).n1).comy, node2(link2(i).n1).comz];
        n2_sub = [node2(link2(i).n2).comx, node2(link2(i).n2).comy, node2(link2(i).n2).comz];
        
        p1 = n1_sub + (n2_sub - n1_sub) * 1/3;
        p2 = n1_sub + (n2_sub - n1_sub) * 2/3;
        
        link2(i).r1 = cal_radius(p1, image);
        link2(i).r2 = cal_radius(p2, image);
    end
    
    % Update node radius features from links
    for i = 1:length(node2)
        for j = 1:length(node2(i).links)
            link_idx = node2(i).links(j);
            node2(i).r1(j) = link2(link_idx).r1;
            node2(i).r2(j) = link2(link_idx).r2;
        end
    end
end

function link2 = add_direction_vectors(link2, node2)
    % Add direction vectors to links
    for i = 1:length(link2)
        start_point = [node2(link2(i).n1).comx, node2(link2(i).n1).comy, node2(link2(i).n1).comz];
        end_point = [node2(link2(i).n2).comx, node2(link2(i).n2).comy, node2(link2(i).n2).comz];
        direction_vector = end_point - start_point;
        link2(i).direction_vector = direction_vector / norm(direction_vector);
        link2(i).len = norm(direction_vector);
    end
end

function node2 = add_curvature_features(node2, link2)
    % Add curvature features to nodes
    for i = 1:length(node2)
        if length(node2(i).links) >= 2
            link1_dir = link2(node2(i).links(1)).direction_vector;
            link2_dir = link2(node2(i).links(2)).direction_vector;
            node2(i).curvature = acos(dot(link1_dir, link2_dir)) / pi;
        else
            node2(i).curvature = 0;
        end
    end
end

function node2 = add_degree_features(node2)
    % Add degree features to nodes
    for i = 1:length(node2)
        node2(i).degree = length(node2(i).links);
    end
end

function link2 = add_resistance_features(link2)
    % Add resistance features to links
    viscosity = 0.04; % Blood viscosity
    for i = 1:length(link2)
        radius = (link2(i).r1 + link2(i).r2)/2;
        resistance = (8 * viscosity * link2(i).len) / (pi * radius^4);
        link2(i).resistance = resistance;
    end
end

function [node2, link2] = add_trachea_features(node2, link2, trachea_nodes, trachea_links, w, l, h)
    % Add trachea proximity features
    D = 15; % Distance threshold (mm)
    L = 0.5; % Minimum sine value
    
    for i = 1:length(node2)
        node2(i).trachea_distance = zeros(3, length(node2(i).links));
        node2(i).trachea_direction = zeros(3, length(node2(i).links));
        node2(i).airway_weight = zeros(3, length(node2(i).links));
    end
    
    for i = 1:length(link2)
        link2(i).trachea_distance = zeros(1, 3);
        link2(i).trachea_direction = zeros(1, 3);
        link2(i).airway_weight = zeros(1, 3);
    end
    
    % Calculate midpoint for each vascular link
    for i = 1:length(link2)
        vascular_midpoint = calculate_midpoint(node2, link2(i), w, l, h);
        
        % Calculate distances to all trachea links
        distances = arrayfun(@(x) norm(vascular_midpoint - ...
            calculate_midpoint(trachea_nodes, trachea_links(x), w, l, h)), ...
            1:length(trachea_links));
        
        % Find closest 3 trachea links
        [~, sorted_idx] = sort(distances);
        closest_idx = sorted_idx(1:min(3, length(sorted_idx)));
        
        % Calculate features for each close trachea link
        for k = 1:length(closest_idx)
            j = closest_idx(k);
            trachea_midpoint = calculate_midpoint(trachea_nodes, trachea_links(j), w, l, h);
            
            % Distance feature
            distance = norm(vascular_midpoint - trachea_midpoint);
            distance_feature = max(D - distance, 0);
            
            % Direction feature
            trachea_dir = [trachea_nodes(trachea_links(j).n2).comx - trachea_nodes(trachea_links(j).n1).comx, ...
                          trachea_nodes(trachea_links(j).n2).comy - trachea_nodes(trachea_links(j).n1).comy, ...
                          trachea_nodes(trachea_links(j).n2).comz - trachea_nodes(trachea_links(j).n1).comz];
            
            vascular_dir = [node2(link2(i).n2).comx - node2(link2(i).n1).comx, ...
                          node2(link2(i).n2).comy - node2(link2(i).n1).comy, ...
                          node2(link2(i).n2).comz - node2(link2(i).n1).comz];
            
            sine_value = norm(cross(trachea_dir, vascular_dir)) / ...
                (norm(trachea_dir) * norm(vascular_dir));
            direction_feature = min(max(sine_value, 0.01), L);
            
            % Airway weight
            airway_weight = distance_feature / direction_feature;
            
            % Update link features
            link2(i).trachea_distance(k) = distance_feature;
            link2(i).trachea_direction(k) = direction_feature;
            link2(i).airway_weight(k) = airway_weight;
            
            % Update connected nodes' features
            update_node_trachea_features(node2, link2(i).n1, i, k, distance_feature, direction_feature, airway_weight);
            update_node_trachea_features(node2, link2(i).n2, i, k, distance_feature, direction_feature, airway_weight);
        end
    end
end

function update_node_trachea_features(node2, node_idx, link_idx, feature_idx, distance, direction, weight)
    % Helper to update node trachea features
    link_pos = find(node2(node_idx).links == link_idx);
    if ~isempty(link_pos)
        node2(node_idx).trachea_distance(feature_idx, link_pos) = distance;
        node2(node_idx).trachea_direction(feature_idx, link_pos) = direction;
        node2(node_idx).airway_weight(feature_idx, link_pos) = weight;
    end
end

function midpoint = calculate_midpoint(node, link, w, l, h)
    % Calculate midpoint between two nodes
    if length(node(link.n1).idx) > 1
        start_point = [node(link.n1).comx, node(link.n1).comy, node(link.n1).comz];
    else
        [x, y, z] = ind2sub([w, l, h], node(link.n1).idx);
        start_point = [x, y, z];
    end
    
    if length(node(link.n2).idx) > 1
        end_point = [node(link.n2).comx, node(link.n2).comy, node(link.n2).comz];
    else
        [x, y, z] = ind2sub([w, l, h], node(link.n2).idx);
        end_point = [x, y, z];
    end
    
    midpoint = (start_point + end_point) / 2;
end

function link2 = add_geometric_deviation_features(link2, node2, mlink2, w, l, h)
    % Add geometric deviation features
    for i = 1:length(link2)
        % Get simplified edge points
        n1_point = [node2(link2(i).n1).comx, node2(link2(i).n1).comy, node2(link2(i).n1).comz];
        n2_point = [node2(link2(i).n2).comx, node2(link2(i).n2).comy, node2(link2(i).n2).comz];
        
        % Get original edge points
        real_points = mlink2(i).point;
        num_points = length(real_points);
        total_deviation = 0;
        
        % Calculate deviation for each original point
        for j = 1:num_points
            [x, y, z] = ind2sub([w, l, h], real_points(j));
            real_point = [x, y, z];
            
            % Calculate shortest distance from point to simplified edge
            line_vec = n2_point - n1_point;
            point_vec = real_point - n1_point;
            line_len = norm(line_vec);
            t = max(0, min(1, dot(point_vec, line_vec) / (line_len^2)));
            nearest = n1_point + t * line_vec;
            dist = norm(real_point - nearest);
            total_deviation = total_deviation + dist;
        end
        
        % Store average deviation
        link2(i).geometric_deviation = total_deviation / num_points;
        link2(i).average_deviation = total_deviation / num_points;
    end
end

function link2 = add_direction_deviation_features(link2, node2, mlink2, w, l, h)
    % Add direction deviation features
    for i = 1:length(link2)
        % Simplified edge direction
        simplified_dir = [node2(link2(i).n2).comx - node2(link2(i).n1).comx, ...
                        node2(link2(i).n2).comy - node2(link2(i).n1).comy, ...
                        node2(link2(i).n2).comz - node2(link2(i).n1).comz];
        simplified_dir = simplified_dir / norm(simplified_dir);
        
        % Original edge direction (first to last point)
        real_points = mlink2(i).point;
        [x1, y1, z1] = ind2sub([w, l, h], real_points(1));
        [x2, y2, z2] = ind2sub([w, l, h], real_points(end));
        real_dir = [x2-x1, y2-y1, z2-z1];
        real_dir = real_dir / norm(real_dir);
        
        % Calculate cosine similarity and angle deviation
        cosine_sim = dot(simplified_dir, real_dir);
        angle_dev = acos(min(max(cosine_sim, -1), 1)) * 180/pi; % Clamp to valid range
        
        link2(i).cosine_similarity = cosine_sim;
        link2(i).angle_deviation = angle_dev;
    end
end

function link2 = add_topological_deviation_features(link2, node2, mlink2)
    % Add topological deviation features
    for i = 1:length(link2)
        % Simplified edge nodes
        n1 = link2(i).n1;
        n2 = link2(i).n2;
        
        % Original edge nodes
        real_n1 = mlink2(i).n1;
        real_n2 = mlink2(i).n2;
        
        % Connectivity deviation (0 if same nodes, 1 otherwise)
        link2(i).connectivity_deviation = ~((n1 == real_n1 && n2 == real_n2) || (n1 == real_n2 && n2 == real_n1));
        
        % Branch structure deviation (difference in node degrees)
        simplified_degrees = length(node2(n1).links) + length(node2(n2).links);
        real_degrees = length(node2(real_n1).links) + length(node2(real_n2).links);
        link2(i).branch_deviation = abs(simplified_degrees - real_degrees);
    end
end

function link2 = add_combined_deviation_features(link2)
    % Add combined deviation metric
    geometric_weight = 0.5;
    direction_weight = 0.3;
    topology_weight = 0.2;
    
    for i = 1:length(link2)
        link2(i).total_deviation = ...
            geometric_weight * link2(i).geometric_deviation + ...
            direction_weight * link2(i).angle_deviation + ...
            topology_weight * link2(i).branch_deviation;
    end
end

function radius = cal_radius(point, image)
    % CAL_RADIUS Calculate radius at a point in the image
    % This is a placeholder - implement actual radius calculation
    radius = 1.0; % Default value
end
