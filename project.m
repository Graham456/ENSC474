function project()
    fprintf('======================================================\n');
    fprintf('STARTING TITAN PIPELINE: ADVANCED MEDICAL SEGMENTATION\n');
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
    % Smooths flat regions (noise) but stops diffusion at edges
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
    % Medical lesions are typically circular/blob-like. The Hessian matrix 
    % identifies these specific shapes natively.
    blob_map = hessian_blob_enhancement(img_enhanced);
    
    % =========================================================================
    % PHASE 5: TISSUE QUANTIZATION (K-MEANS APPROXIMATION)
    % =========================================================================
    fprintf('[5/10] Multi-level Otsu Tissue Quantization...\n');
    % Instead of binary thresholding, we separate the image into 3 classes:
    % 1: Darkest (Often Lesions/Shadows), 2: Mid (Healthy Tissue), 3: Bright (Fat/Skin)
    num_classes = 3;
    levels = multithresh(img_enhanced, num_classes - 1);
    quantized_img = imquantize(img_enhanced, levels);
    
    % Isolate the darkest class (Class 1) as our primary tumor candidates
    dark_tissue_mask = (quantized_img == 1);
    
    % =========================================================================
    % PHASE 6: MORPHOLOGICAL RECONSTRUCTION (TOPOLOGY CLEANUP)
    % =========================================================================
    fprintf('[6/10] Morphological Opening by Reconstruction...\n');
    % Erode to find the absolute core of the regions, then reconstruct to 
    % regain the shape without including the noise.
    marker_cores = imerode(dark_tissue_mask, strel('disk', 4));
    clean_markers = imreconstruct(marker_cores, dark_tissue_mask);
    
    % Remove highly irrelevant small artifacts
    clean_markers = bwareaopen(clean_markers, 40);
    
    % =========================================================================
    % PHASE 7: SMOOTHED GRADIENT VECTOR FIELDS
    % =========================================================================
    fprintf('[7/10] Calculating Smoothed Gradient Vector Fields...\n');
    [Gx, Gy] = imgradientxy(img_enhanced, 'sobel');
    gradient_mag = sqrt(Gx.^2 + Gy.^2);
    
    % Blur the gradient so the edges form solid, unbroken walls for the watershed
    H_smooth = fspecial('gaussian', [11 11], 3);
    smooth_gradient = imfilter(gradient_mag, H_smooth, 'replicate');
    smooth_gradient = mat2gray(smooth_gradient);
    
    % =========================================================================
    % PHASE 8: MARKER-CONTROLLED WATERSHED
    % =========================================================================
    fprintf('[8/10] Executing Marker-Controlled Watershed...\n');
    % Define background markers (pixels furthest from our clean_markers)
    D = bwdist(clean_markers);
    DL = watershed(D);
    bg_markers = (DL == 0);
    
    % Impose topological minima on the gradient image at our markers
    grad_imposed = imimposemin(smooth_gradient, clean_markers | bg_markers);
    
    % Run watershed: regions will expand from clean_markers and stop at gradients
    watershed_labels = watershed(grad_imposed);
    rough_segments = (watershed_labels > 0) & ~bg_markers;
    
    % =========================================================================
    % PHASE 9: MULTI-FACTOR HEURISTIC REGION SELECTION
    % =========================================================================
    fprintf('[9/10] Scoring Regions via Medical Heuristics...\n');
    % Evaluates all watershed segments and picks the one that is most mathematically
    % "tumor-like" based on Size, Solidity, Intensity, and Hessian Blobness.
    best_candidate_mask = select_ultimate_lesion(rough_segments, img_enhanced, blob_map);
    
    % =========================================================================
    % PHASE 10: TWO-PASS ACTIVE CONTOURS (SNAKES)
    % =========================================================================
    fprintf('[10/10] Deploying Two-Pass Active Contours...\n');
    if sum(best_candidate_mask(:)) > 0
        % Pass 1: Region-based (Chan-Vese) to secure the overall mass
        % Rubber-band effect (positive contraction) to prevent ballooning
        snake_pass1 = activecontour(img_enhanced, best_candidate_mask, 100, 'Chan-Vese', ...
            'SmoothFactor', 2.0, 'ContractionBias', 0.1);
            
        % Pass 2: Edge-based to snap exactly to the highest gradient boundaries
        finalMask = activecontour(smooth_gradient, snake_pass1, 50, 'edge', ...
            'SmoothFactor', 1.0, 'ContractionBias', 0.1);
    else
        fprintf('WARNING: No valid lesion candidate found.\n');
        finalMask = false(size(img));
    end
    
    % Final Polish
    finalMask = imfill(finalMask, 'holes');
    finalMask = imclose(finalMask, strel('disk', 3));

    % =========================================================================
    % EVALUATION & METRICS
    % =========================================================================
    fprintf('\n======================================================\n');
    fprintf('PIPELINE COMPLETE. CALCULATING METRICS...\n');
    fprintf('======================================================\n');
    
    gt_resized = imresize(gt, [size(img,1), size(img,2)], 'nearest');
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
    
    % percentile clipping to remove extreme dead pixels / artifact spikes
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
    % Calculate 2nd order spatial derivatives using smoothed Gaussian kernels
    sigma = 2.0;
    [X, Y] = meshgrid(-round(3*sigma):round(3*sigma));
    
    % Gaussian kernels
    G = exp(-(X.^2 + Y.^2) / (2*sigma^2)) / (2*pi*sigma^2);
    
    % 2nd derivative kernels
    Dxx = G .* (X.^2/sigma^4 - 1/sigma^2);
    Dyy = G .* (Y.^2/sigma^4 - 1/sigma^2);
    Dxy = G .* (X.*Y/sigma^4);
    
    % Filter image
    Ixx = imfilter(img, Dxx, 'replicate');
    Iyy = imfilter(img, Dyy, 'replicate');
    Ixy = imfilter(img, Dxy, 'replicate');
    
    % Calculate Eigenvalues of the Hessian Matrix
    % lambda1 is the principal eigenvalue (strongest curvature)
    % lambda2 is the secondary eigenvalue
    tmp = sqrt((Ixx - Iyy).^2 + 4*Ixy.^2);
    lambda1 = 0.5 * (Ixx + Iyy + tmp);
    lambda2 = 0.5 * (Ixx + Iyy - tmp);
    
    % Blobness filter: High when both eigenvalues are large and have the same sign
    % (We take absolute values to detect both dark and bright blobs, 
    % though we filter for dark ones later).
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
        % 1. Area Score: Punish tiny specks and massive background chunks
        area_ratio = stats(i).Area / (size(originalImg,1) * size(originalImg,2));
        if area_ratio < 0.005 || area_ratio > 0.4
            areaScore = 0; % Disqualify extremes
        else
            areaScore = stats(i).Area / maxArea;
        end
        
        % 2. Shape Score: Tumors are usually solid masses, not squiggles
        if isfield(stats, 'Circularity') && ~isempty(stats(i).Circularity)
            shapeScore = stats(i).Circularity;
        else
            shapeScore = stats(i).Solidity;
        end
        
        % 3. Intensity Score: Tumors are usually darker than surrounding tissue
        meanIntensity = mean(originalImg(stats(i).PixelIdxList));
        contrastScore = 1 - meanIntensity; 
        
        % 4. Hessian Score: Does it register mathematically as a "blob"?
        meanBlobness = mean(blob_map(stats(i).PixelIdxList));
        
        % 5. Boundary Location Penalty: Tumors rarely touch the exact borders perfectly
        bb = stats(i).BoundingBox;
        touchesBorder = (bb(1) <= 2) || (bb(2) <= 2) || ...
                        ((bb(1) + bb(3)) >= size(originalImg,2)-2) || ...
                        ((bb(2) + bb(4)) >= size(originalImg,1)-2);
        borderPenalty = 1.0;
        if touchesBorder
            borderPenalty = 0.5; % Cut score in half if it hits the image edge
        end
        
        % Fusion Formula
        scores(i) = ((shapeScore * 0.3) + (contrastScore * 0.3) + ...
                     (meanBlobness * 0.2) + (areaScore * 0.2)) * borderPenalty;
    end
    
    [~, bestIdx] = max(scores);
    finalMask = (L == bestIdx);
end