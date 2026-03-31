function project_alltestcases()
    fprintf('======================================================\n');
    fprintf('STARTING TITAN PIPELINE: BATCH PROCESSING MODE\n');
    fprintf('======================================================\n\n');
    
    % Test cases for batch processing
    test_cases = {
        'TEST0.jpeg', 'TEST0GT.jpeg';
        'TEST1.jpeg', 'TEST1GT.jpeg';
        'TEST2.jpeg', 'TEST2GT.jpeg';
        'TEST3.jpeg', 'TEST3GT.jpeg'
    };
    
    num_cases = size(test_cases, 1);
    all_iou = zeros(num_cases, 1);
    all_dice = zeros(num_cases, 1);
    
    % Prepare Popup 1: The Master Summary Grid
    fig_summary = figure('Name', 'Popup 1: All Test Cases (Final Results)', ...
                         'NumberTitle', 'off', 'Position', [50, 50, 1400, 950]);
    
    % Store diagnostic phases for Popup 2
    diagnostic_phases = struct();
    
    for i = 1:num_cases
        imgName = test_cases{i, 1};
        gtName  = test_cases{i, 2};
        
        fprintf('Processing Case %d/%d: [%s]...\n', i, num_cases, imgName);
        
        if ~exist(imgName, 'file') || ~exist(gtName, 'file')
            fprintf('  -> ERROR: Files %s or %s not found. Skipping.\n', imgName, gtName);
            continue;
        end
        
        % Run the core algorithm on this specific image
        [img, gt_resized, finalMask, metrics, phases] = process_single_image(imgName, gtName);
        
        % Store metrics
        all_iou(i) = metrics.iou;
        all_dice(i) = metrics.dice;
        
        fprintf('  -> IoU: %.4f | Dice: %.4f | Sens: %.4f | Spec: %.4f\n', ...
            metrics.iou, metrics.dice, metrics.sensitivity, metrics.specificity);
            
        % Save the phases of the first image (TEST0) for Popup 2
        if i == 1
            diagnostic_phases.img = img;
            diagnostic_phases.gt = gt_resized;
            diagnostic_phases.finalMask = finalMask;
            diagnostic_phases.data = phases;
            diagnostic_phases.name = imgName;
        end
        
        % Render this case into Popup 1 (The Summary Grid)
        figure(fig_summary);
        
        % 1. Original Image
        subplot(num_cases, 3, (i-1)*3 + 1);
        imshow(img);
        if i == 1, title('Original Input', 'FontWeight', 'bold'); end
        ylabel(strrep(imgName, '.jpeg', ''), 'FontWeight', 'bold', 'Interpreter', 'none');
        
        % 2. Ground Truth
        subplot(num_cases, 3, (i-1)*3 + 2);
        imshow(gt_resized);
        if i == 1, title('Ground Truth Mask', 'FontWeight', 'bold'); end
        
        % 3. Final Overlay
        subplot(num_cases, 3, (i-1)*3 + 3);
        imshow(img);
        hold on;
        visboundaries(finalMask, 'Color', 'r', 'LineWidth', 2);
        visboundaries(gt_resized, 'Color', 'g', 'LineWidth', 1, 'LineStyle', '--');
        hold off;
        if i == 1, title('Red: Algorithm | Green: Truth', 'FontWeight', 'bold'); end
    end
    
    % Calculate and Output Final Batch Statistics
    mean_iou = mean(all_iou(all_iou > 0)); % Ignore skipped files if any
    mean_dice = mean(all_dice(all_dice > 0));
    
    fprintf('\n======================================================\n');
    fprintf('BATCH PROCESSING COMPLETE\n');
    fprintf('======================================================\n');
    fprintf('Total Images Processed: %d\n', sum(all_iou > 0));
    fprintf('MEAN IoU SCORE:         %.4f\n', mean_iou);
    fprintf('MEAN DICE SCORE:        %.4f\n', mean_dice);
    fprintf('======================================================\n');
    
    % Render Popup 2: The Phase-by-Phase Diagnostic (For TEST0)
    render_diagnostic_dashboard(diagnostic_phases);
end

% =========================================================================
% CORE PIPELINE (RUNS ONCE PER IMAGE)
% =========================================================================
function [img, gt_resized, finalMask, metrics, phases] = process_single_image(imgName, gtName)
    
    [img, gt] = preprocess_data(imgName, gtName);
    phases = struct(); 
    
    % Phase 1 & 2: Denoising and Anisotropic Diffusion
    phases.img_freq = butterworth_lowpass(img, 45, 2);
    phases.img_diffused = anisodiff2D(phases.img_freq, 20, 0.15, 25, 1);
    
    % Phase 3: Contrast Enhancement
    phases.img_enhanced = adapthisteq(phases.img_diffused, 'ClipLimit', 0.015, 'Distribution', 'rayleigh');
    
    % Phase 4: Hessian Blob Detection
    phases.blob_map = hessian_blob_enhancement(phases.img_enhanced);
    
    % Phase 5: Multi-level K-Means
    num_classes = 3;
    levels = multithresh(phases.img_enhanced, num_classes - 1);
    quantized_img = imquantize(phases.img_enhanced, levels);
    phases.dark_tissue_mask = (quantized_img == 1);
    
    % Phase 6: Morphological Cleanup
    marker_cores = imerode(phases.dark_tissue_mask, strel('disk', 4));
    clean_markers = imreconstruct(marker_cores, phases.dark_tissue_mask);
    phases.clean_markers = bwareaopen(clean_markers, 40);
    
    % Phase 7: Smoothed Gradients
    [Gx, Gy] = imgradientxy(phases.img_enhanced, 'sobel');
    gradient_mag = sqrt(Gx.^2 + Gy.^2);
    H_smooth = fspecial('gaussian', [11 11], 3);
    phases.smooth_gradient = mat2gray(imfilter(gradient_mag, H_smooth, 'replicate'));
    
    % Phase 8: Marker-Controlled Watershed
    D = bwdist(phases.clean_markers);
    DL = watershed(D);
    bg_markers = (DL == 0);
    grad_imposed = imimposemin(phases.smooth_gradient, phases.clean_markers | bg_markers);
    phases.watershed_labels = watershed(grad_imposed);
    rough_segments = (phases.watershed_labels > 0) & ~bg_markers;
    
    % Phase 9: Heuristic Selection (JPEG border penalty removed)
    phases.best_candidate_mask = select_ultimate_lesion(rough_segments, phases.img_enhanced, phases.blob_map);
    
    % Phase 10: Robust Active Contours
    if sum(phases.best_candidate_mask(:)) > 0
        finalMask = activecontour(phases.img_enhanced, phases.best_candidate_mask, 150, 'Chan-Vese', ...
            'SmoothFactor', 2.0, 'ContractionBias', 0.05);
    else
        finalMask = false(size(img));
    end
    
    % ========================================================
    % THE BRIDGE BREAKER: Sever thin JPEG artifact connections
    % ========================================================
    finalMask = imopen(finalMask, strel('disk', 5));
    
    % ========================================================
    % THE SINGLE LESION RULE: Keep only the largest contiguous mass
    % ========================================================
    [L_final, num_final] = bwlabel(finalMask);
    if num_final > 1
        stats_final = regionprops(L_final, 'Area');
        [~, maxIdx] = max([stats_final.Area]);
        finalMask = (L_final == maxIdx);
    end
    
    % Final Polish
    finalMask = imclose(finalMask, strel('disk', 6));
    finalMask = imfill(finalMask, 'holes');
    
    % Metrics Calculation
    gt_resized = imresize(gt, [size(img,1), size(img,2)], 'nearest');
    intersection = sum(finalMask(:) & gt_resized(:));
    unionArea = sum(finalMask(:) | gt_resized(:));
    
    metrics.iou = intersection / max(unionArea, 1);
    metrics.dice = 2 * intersection / max(sum(finalMask(:)) + sum(gt_resized(:)), 1);
    metrics.fp = sum(finalMask(:) & ~gt_resized(:));
    metrics.fn = sum(~finalMask(:) & gt_resized(:));
    metrics.specificity = sum(~finalMask(:) & ~gt_resized(:)) / max(sum(~gt_resized(:)), 1);
    metrics.sensitivity = intersection / max(sum(gt_resized(:)), 1);
end

% =========================================================================
% HEURISTIC SELECTION (FIXED FOR JPEG LEAKS)
% =========================================================================
function finalMask = select_ultimate_lesion(watershed_mask, originalImg, blob_map)
    [L, num] = bwlabel(watershed_mask);
    if num == 0
        finalMask = watershed_mask; 
        return; 
    end
    
    stats = regionprops(L, 'Area', 'Solidity', 'Perimeter', 'PixelIdxList');
    scores = zeros(num, 1);
    maxArea = max([stats.Area]);
    
    for i = 1:num
        % 1. Size
        areaScore = stats(i).Area / maxArea;
        
        % 2. Shape
        if isfield(stats, 'Circularity') && ~isempty(stats(i).Circularity)
            shapeScore = stats(i).Circularity;
        else
            shapeScore = stats(i).Solidity;
        end
        
        % 3. Contrast
        meanIntensity = mean(originalImg(stats(i).PixelIdxList));
        contrastScore = 1 - meanIntensity; 
        
        % 4. Blobness
        meanBlobness = mean(blob_map(stats(i).PixelIdxList));
        
        % FUSION LOGIC (Border penalty fully removed)
        scores(i) = (areaScore * 0.45) + (contrastScore * 0.25) + ...
                    (shapeScore * 0.20) + (meanBlobness * 0.10);
    end
    
    [~, bestIdx] = max(scores);
    finalMask = (L == bestIdx);
end

% =========================================================================
% POPUP 2: DIAGNOSTIC DASHBOARD RENDERER
% =========================================================================
function render_diagnostic_dashboard(dp)
    if isempty(fieldnames(dp)), return; end
    
    figure('Name', sprintf('Popup 2: Mathematical Phases (%s)', dp.name), ...
           'NumberTitle', 'off', 'Position', [100, 100, 1400, 800]);
           
    subplot(3, 4, 1); imshow(dp.img); title('1. Original Input');
    subplot(3, 4, 2); imshow(dp.data.img_freq); title('2. Butterworth Filter');
    subplot(3, 4, 3); imshow(dp.data.img_diffused); title('3. Aniso Diffusion');
    subplot(3, 4, 4); imshow(dp.data.img_enhanced); title('4. Enhanced Contrast');
    
    subplot(3, 4, 5); imshow(dp.data.blob_map); colormap(gca, jet); title('5. Hessian Map');
    subplot(3, 4, 6); imshow(dp.data.dark_tissue_mask); title('6. Quantized Tissue');
    subplot(3, 4, 7); imshow(dp.data.clean_markers); title('7. Cleaned Markers');
    subplot(3, 4, 8); imshow(dp.data.smooth_gradient); title('8. Smooth Gradients');
    
    subplot(3, 4, 9); imshow(label2rgb(dp.data.watershed_labels, 'jet', 'k', 'shuffle')); title('9. Watershed Basins');
    subplot(3, 4, 10); imshow(dp.data.best_candidate_mask); title('10. Selected Region');
    
    subplot(3, 4, [11, 12]); 
    imshow(dp.img); 
    title('11 & 12. Final Snake Contour (Red) vs GT (Green)');
    hold on;
    visboundaries(dp.finalMask, 'Color', 'r', 'LineWidth', 2);
    visboundaries(dp.gt, 'Color', 'g', 'LineWidth', 1, 'LineStyle', '--');
    hold off;
end

% =========================================================================
% HELPER MATH FUNCTIONS
% =========================================================================
function [imgNorm, gtMask] = preprocess_data(imgP, mskP)
    rawImg = im2double(imread(imgP));
    rawGT  = im2double(imread(mskP));
    if size(rawImg, 3) == 3, rawImg = rgb2gray(rawImg); end
    if size(rawGT,  3) == 3, rawGT  = rgb2gray(rawGT);  end
    
    lower_bound = quantile(rawImg(:), 0.02);
    upper_bound = quantile(rawImg(:), 0.98);
    clipped = max(lower_bound, min(upper_bound, rawImg));
    imgNorm = (clipped - lower_bound) / (upper_bound - lower_bound);
    gtMask = rawGT > 0.5;
end

function output = butterworth_lowpass(img, cutoff, order)
    [M, N] = size(img);
    [U, V] = meshgrid(1:N, 1:M);
    centerX = floor(N/2) + 1;
    centerY = floor(M/2) + 1;
    D = sqrt((U - centerX).^2 + (V - centerY).^2);
    H = 1 ./ (1 + (D / cutoff).^(2*order));
    F = fftshift(fft2(img));
    G = F .* H;
    output = mat2gray(real(ifft2(ifftshift(G))));
end

function imgOut = anisodiff2D(img, niter, lambda, kappa, option)
    imgOut = img;
    for i = 1:niter
        deltaN = circshift(imgOut, [-1  0]) - imgOut;
        deltaS = circshift(imgOut, [ 1  0]) - imgOut;
        deltaE = circshift(imgOut, [ 0  1]) - imgOut;
        deltaW = circshift(imgOut, [ 0 -1]) - imgOut;
        
        if option == 1
            cN = exp(-(deltaN/kappa).^2); cS = exp(-(deltaS/kappa).^2);
            cE = exp(-(deltaE/kappa).^2); cW = exp(-(deltaW/kappa).^2);
        else
            cN = 1 ./ (1 + (deltaN/kappa).^2); cS = 1 ./ (1 + (deltaS/kappa).^2);
            cE = 1 ./ (1 + (deltaE/kappa).^2); cW = 1 ./ (1 + (deltaW/kappa).^2);
        end
        imgOut = imgOut + lambda * (cN.*deltaN + cS.*deltaS + cE.*deltaE + cW.*deltaW);
    end
end

function blob_map = hessian_blob_enhancement(img)
    sigma = 2.0;
    [X, Y] = meshgrid(-round(3*sigma):round(3*sigma));
    G = exp(-(X.^2 + Y.^2) / (2*sigma^2)) / (2*pi*sigma^2);
    Dxx = G .* (X.^2/sigma^4 - 1/sigma^2);
    Dyy = G .* (Y.^2/sigma^4 - 1/sigma^2);
    Dxy = G .* (X.*Y/sigma^4);
    
    Ixx = imfilter(img, Dxx, 'replicate');
    Iyy = imfilter(img, Dyy, 'replicate');
    Ixy = imfilter(img, Dxy, 'replicate');
    
    tmp = sqrt((Ixx - Iyy).^2 + 4*Ixy.^2);
    lambda1 = 0.5 * (Ixx + Iyy + tmp);
    lambda2 = 0.5 * (Ixx + Iyy - tmp);
    blob_map = mat2gray(abs(lambda1) .* abs(lambda2));
end