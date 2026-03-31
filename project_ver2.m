function project()
    fprintf('======================================================\n');
    fprintf('STARTING PIPELINE: ADVANCED MEDICAL SEGMENTATION\n');
    fprintf('======================================================\n');
    
    imagePath = 'projectimg.png';
    maskPath  = 'GTmask.png';
    
    if ~exist(imagePath, 'file') || ~exist(maskPath, 'file')
        error('CRITICAL: Image or GTmask not found in the current directory.');
    end
    
    [img, gt] = preprocess_data(imagePath, maskPath);
    fprintf('Image loaded successfully: %d x %d pixels\n', size(img,1), size(img,2));
    
    % =========================================================================
    % PHASE 1: FREQUENCY DOMAIN DENOISING
    % =========================================================================
    fprintf('\n[1/10] Applying Butterworth Low-Pass Frequency Filter...\n');
    img_freq = butterworth_lowpass(img, 45, 2);
    
    % =========================================================================
    % PHASE 2: ANISOTROPIC DIFFUSION (EDGE-PRESERVING SMOOTHING)
    % =========================================================================
    fprintf('[2/10] Solving Perona-Malik Anisotropic Diffusion PDEs...\n');
    img_diffused = anisodiff2D(img_freq, 20, 0.15, 25, 1);
    
    % =========================================================================
    % PHASE 3: ADAPTIVE CONTRAST ENHANCEMENT
    % =========================================================================
    fprintf('[3/10] Applying Contrast-Limited Adaptive Histogram Equalization...\n');
    img_enhanced = adapthisteq(img_diffused, 'ClipLimit', 0.015, 'Distribution', 'rayleigh');
    
    % =========================================================================
    % PHASE 4: HESSIAN MATRIX BLOB DETECTION (2nd Order Derivatives)
    % =========================================================================
    fprintf('[4/10] Computing Hessian Eigenvalues for Blob Enhancement...\n');
    blob_map = hessian_blob_enhancement(img_enhanced);
    
    % =========================================================================
    % PHASE 5: TISSUE QUANTIZATION (K-MEANS APPROXIMATION)
    % =========================================================================
    fprintf('[5/10] Multi-level Otsu Tissue Quantization...\n');
    num_classes = 3;
    levels = multithresh(img_enhanced, num_classes - 1);
    quantized_img = imquantize(img_enhanced, levels);
    dark_tissue_mask = (quantized_img == 1);
    
    % =========================================================================
    % PHASE 6: MORPHOLOGICAL RECONSTRUCTION (TOPOLOGY CLEANUP)
    % =========================================================================
    fprintf('[6/10] Morphological Opening by Reconstruction...\n');
    marker_cores = imerode(dark_tissue_mask, strel('disk', 4));
    clean_markers = imreconstruct(marker_cores, dark_tissue_mask);
    clean_markers = bwareaopen(clean_markers, 40);
    
    % =========================================================================
    % PHASE 7: SMOOTHED GRADIENT VECTOR FIELDS
    % =========================================================================
    fprintf('[7/10] Calculating Smoothed Gradient Vector Fields...\n');
    [Gx, Gy] = imgradientxy(img_enhanced, 'sobel');
    gradient_mag = sqrt(Gx.^2 + Gy.^2);
    
    H_smooth = fspecial('gaussian', [11 11], 3);
    smooth_gradient = imfilter(gradient_mag, H_smooth, 'replicate');
    smooth_gradient = mat2gray(smooth_gradient);
    
    % =========================================================================
    % PHASE 8: MARKER-CONTROLLED WATERSHED
    % =========================================================================
    fprintf('[8/10] Executing Marker-Controlled Watershed...\n');
    D = bwdist(clean_markers);
    DL = watershed(D);
    bg_markers = (DL == 0);
    
    grad_imposed = imimposemin(smooth_gradient, clean_markers | bg_markers);
    watershed_labels = watershed(grad_imposed);
    rough_segments = (watershed_labels > 0) & ~bg_markers;
    
    % =========================================================================
    % PHASE 9: MULTI-FACTOR HEURISTIC REGION SELECTION
    % =========================================================================
    fprintf('[9/10] Scoring Regions via Medical Heuristics...\n');
    best_candidate_mask = select_ultimate_lesion(rough_segments, img_enhanced, blob_map);
    
    % =========================================================================
    % PHASE 10: TWO-PASS ACTIVE CONTOURS (SNAKES)
    % =========================================================================
    fprintf('[10/10] Deploying Two-Pass Active Contours...\n');
    if sum(best_candidate_mask(:)) > 0
        snake_pass1 = activecontour(img_enhanced, best_candidate_mask, 100, 'Chan-Vese', ...
            'SmoothFactor', 2.0, 'ContractionBias', 0.1);
            
        finalMask = activecontour(smooth_gradient, snake_pass1, 50, 'edge', ...
            'SmoothFactor', 1.0, 'ContractionBias', 0.1);
    else
        fprintf('WARNING: No valid lesion candidate found.\n');
        finalMask = false(size(img));
    end
    
    finalMask = imfill(finalMask, 'holes');
    finalMask = imclose(finalMask, strel('disk', 3));

    % =========================================================================
    % VISUAL DIAGNOSTIC DASHBOARD
    % =========================================================================
    fprintf('\n[Visuals] Generating Phase-by-Phase Diagnostic Dashboard...\n');
    
    figure('Name', 'Titan Pipeline: Visual Diagnostic', 'NumberTitle', 'off', 'Position', [50, 50, 1600, 900]);
    
    subplot(3, 4, 1); imshow(img); title('1. Original Image');
    subplot(3, 4, 2); imshow(img_freq); title('2. Butterworth (Phase 1)');
    subplot(3, 4, 3); imshow(img_diffused); title('3. Aniso Diffusion (Phase 2)');
    subplot(3, 4, 4); imshow(img_enhanced); title('4. Enhanced (Phase 3)');
    
    subplot(3, 4, 5); imshow(blob_map); colormap(gca, jet); title('5. Hessian Map (Phase 4)');
    subplot(3, 4, 6); imshow(dark_tissue_mask); title('6. Quantized (Phase 5)');
    subplot(3, 4, 7); imshow(clean_markers); title('7. Markers (Phase 6)');
    subplot(3, 4, 8); imshow(smooth_gradient); title('8. Smooth Gradients (Phase 7)');
    
    subplot(3, 4, 9); imshow(label2rgb(watershed_labels, 'jet', 'k', 'shuffle')); title('9. Watershed (Phase 8)');
    subplot(3, 4, 10); imshow(best_candidate_mask); title('10. Selected Core (Phase 9)');
    
    subplot(3, 4, [11, 12]); 
    imshow(img); 
    title('11. Final Snake Contour (Red) vs Ground Truth (Green)');
    hold on;
    visboundaries(finalMask, 'Color', 'r', 'LineWidth', 2);
    
    gt_resized = imresize(gt, [size(img,1), size(img,2)], 'nearest');
    visboundaries(gt_resized, 'Color', 'g', 'LineWidth', 1, 'LineStyle', '--');
    hold off;
    drawnow;

    % =========================================================================
    % EVALUATION & METRICS
    % =========================================================================
    fprintf('\n======================================================\n');
    fprintf('PIPELINE COMPLETE. CALCULATING METRICS...\n');
    fprintf('======================================================\n');
    
    intersection = sum(finalMask(:) & gt_resized(:));
    unionArea = sum(finalMask(:) | gt_resized(:));
    
    iou = intersection / max(unionArea, 1);
    dice = 2 * intersection / max(sum(finalMask(:)) + sum(gt_resized(:)), 1);
    fp = sum(finalMask(:) & ~gt_resized(:));
    fn = sum(~finalMask(:) & gt_resized(:));
    specificity = sum(~finalMask(:) & ~gt_resized(:)) / max(sum(~gt_resized(:)), 1);
    sensitivity = intersection / max(sum(gt_resized(:)), 1);
    
    fprintf('IoU (Intersection over Union): %.4f\n', iou);
    fprintf('Dice Score:                    %.4f\n', dice);
    fprintf('Sensitivity (Recall):          %.4f\n', sensitivity);
    fprintf('Specificity:                   %.4f\n', specificity);
    fprintf('False Positives:               %d\n', fp);
    fprintf('False Negatives:               %d\n', fn);
    fprintf('======================================================\n');
end

% =========================================================================
% --- LOCAL FUNCTIONS / CORE ALGORITHMS ---
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
    output = real(ifft2(ifftshift(G)));
    output = mat2gray(output);
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
    
    blob_map = abs(lambda1) .* abs(lambda2);
    blob_map = mat2gray(blob_map);
end

function finalMask = select_ultimate_lesion(watershed_mask, originalImg, blob_map)
    [L, num] = bwlabel(watershed_mask);
    if num == 0
        finalMask = watershed_mask; 
        return; 
    end
    
    stats = regionprops(L, 'Area', 'Solidity', 'Perimeter', 'PixelIdxList', 'BoundingBox');
    scores = zeros(num, 1);
    
    maxArea = max([stats.Area]);
    
    for i = 1:num
        area_ratio = stats(i).Area / (size(originalImg,1) * size(originalImg,2));
        if area_ratio < 0.005 || area_ratio > 0.4
            areaScore = 0; 
        else
            areaScore = stats(i).Area / maxArea;
        end
        
        if isfield(stats, 'Circularity') && ~isempty(stats(i).Circularity)
            shapeScore = stats(i).Circularity;
        else
            shapeScore = stats(i).Solidity;
        end
        
        meanIntensity = mean(originalImg(stats(i).PixelIdxList));
        contrastScore = 1 - meanIntensity; 
        
        meanBlobness = mean(blob_map(stats(i).PixelIdxList));
        
        bb = stats(i).BoundingBox;
        touchesBorder = (bb(1) <= 2) || (bb(2) <= 2) || ...
                        ((bb(1) + bb(3)) >= size(originalImg,2)-2) || ...
                        ((bb(2) + bb(4)) >= size(originalImg,1)-2);
        borderPenalty = 1.0;
        if touchesBorder
            borderPenalty = 0.5; 
        end
        
        scores(i) = ((shapeScore * 0.3) + (contrastScore * 0.3) + ...
                     (meanBlobness * 0.2) + (areaScore * 0.2)) * borderPenalty;
    end
    
    [~, bestIdx] = max(scores);
    finalMask = (L == bestIdx);
end