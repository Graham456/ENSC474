function project()
    
    imagePath = 'projectimg.png'; % 
    maskPath  = 'GTmask.png';

    % Phase 1: Preprocessing
    [img, gt] = preprocess_data(imagePath, maskPath);
    
    % Place to call each function 
    homo_filter_out = homomorphic_filter(img);
    contrast_out = contrast_enhancement(homo_filter_out);
    featureMap = generate_features(contrast_out);
    binaryMask = segment_image(featureMap);
    cleanMask = refine_mask(binaryMask);
    splitMask = apply_watershed(cleanMask);
    finalMask = select_best_lesion(splitMask);
    
     % DEBUG: show every intermediate stage so we can see where it breaks
    figure('Name', 'Pipeline Debug', 'Position', [100 100 1400 800]);
    
    subplot(2,4,1); imshow(img);            title('1. Raw Input');
    subplot(2,4,2); imshow(homo_filter_out);title('2. Homomorphic');
    subplot(2,4,3); imshow(contrast_out);   title('3. Contrast Enhanced');
    subplot(2,4,4); imshow(featureMap);     title('4. Feature Map');
    subplot(2,4,5); imshow(binaryMask);     title('5. Binary Mask');
    subplot(2,4,6); imshow(cleanMask);      title('6. Refined Mask');
    subplot(2,4,7); imshow(splitMask);      title('7. After Watershed');
    subplot(2,4,8); imshow(finalMask);      title('8. Final vs GT');
    
    % Also print the feature map stats so we can see the value distribution
    fprintf('--- Feature Map Stats ---\n');
    fprintf('Min: %.4f, Max: %.4f\n', min(featureMap(:)), max(featureMap(:)));
    fprintf('Mean: %.4f, Std: %.4f\n', mean(featureMap(:)), std(featureMap(:)));
    fprintf('Pixels above 0.5: %d\n', sum(featureMap(:) > 0.5));
    fprintf('Pixels above 0.7: %d\n', sum(featureMap(:) > 0.7));
    fprintf('Pixels above 0.9: %d\n', sum(featureMap(:) > 0.9));

    % Overlay comparison: red = finalMask, green = ground truth
    figure('Name', 'Overlay Comparison');
    overlayImg = zeros(size(img,1), size(img,2), 3);
    gt_resized = imresize(gt, [size(img,1), size(img,2)], 'nearest');

    overlayImg(:,:,1) = double(finalMask);   % red channel = your detection
    overlayImg(:,:,2) = double(gt_resized);          % green channel = ground truth
    imshow(overlayImg);
    title('Red=finalMask | Green=GT | Yellow=Overlap');

    fprintf('Phase 1 Complete using %s and %s.\n', imagePath, maskPath);
end

% --- LOCAL FUNCTIONS ---

function [imgNorm, gtMask] = preprocess_data(imgP, mskP)
    rawImg = im2double(imread(imgP));
    rawGT  = im2double(imread(mskP));
    if size(rawImg, 3) == 3, rawImg = rgb2gray(rawImg); end % convert to 2D greyscale if images are 2D (RGB)
    if size(rawGT, 3) == 3,  rawGT  = rgb2gray(rawGT);  end
    
    if any(size(rawGT) ~= size(rawImg)) % added when i had to resize GT image to fix error
        rawGT = imresize(rawGT, [size(rawImg,1), size(rawImg,2)], 'nearest');
        fprintf('Warning: GT mask resized from original to match image dimensions.\n');
    end

    lower = quantile(rawImg(:), 0.01); % remove top and bottom 1% to ignore bright text or black borders
    upper = quantile(rawImg(:), 0.99); 
    
    clipped = max(lower, min(upper, rawImg));
    
    imgNorm = (clipped - lower) / (upper - lower); % force intensity to be between 0 and 1
    
    gtMask = rawGT > 0.5; % binarize ground truth (need to use for IoU calc later)
end

function homo_filter_out = homomorphic_filter(img) % step 2, homomorphic filtering for speckle reduction
    
    imgLog = log(1 + img); % avoid using log(0) 
    
    % Perform 2D FFT and shift the zero-frequency component to the center
    F = fftshift(fft2(imgLog));
    
   
    [M, N] = size(img); % high pass filter kernal
    [U, V] = meshgrid(1:N, 1:M);
    
    % Define the center coordinates
    centerX = floor(N/2) + 1;
    centerY = floor(M/2) + 1;
    
    % D is the distance from the center (frequency origin)
    D = sqrt((U - centerX).^2 + (V - centerY).^2);
    
    % Sigma controls the 'cutoff'. 
    % Lower sigma = smoother/blurrier, Higher sigma = sharper/more noise kept.
    sigma = 30; 
    gammaL = 0.5;
    gammaH = 1.5;

    H = (gammaH - gammaL) * (1 - exp(-(D.^2) / (2 * sigma^2))) + gammaL;
    
    % 4. Apply the Filter
    G = F .* H;
    imgFiltered = real(ifft2(ifftshift(G)));
   
    homo_filter_out = exp(imgFiltered) - 1;
    % Standardize the output to [0, 1] for Phase 3 (Contrast Enhancement)
    if max(homo_filter_out(:)) > min(homo_filter_out(:))
        homo_filter_out = (homo_filter_out - min(homo_filter_out(:))) / (max(homo_filter_out(:)) - min(homo_filter_out(:))); % standardize output to [0,1] for Phase 3 contrast enhancement
    end
end

% Phase 3: Contrast enhancement


function contrast_out = contrast_enhancement(homo_filter_out) 
    % --- Step A: Local Contrast Enhancement ---
    % We use a local mean to adjust intensities
    % Define a local window size (e.g., 15x15)
    h = ones(15,15) / (15*15);
    localMean = imfilter(homo_filter_out, h, 'replicate');
    
    % Enhance: Output = Gain * (Input - localMean) + localMean
    % A gain > 1 increases local contrast
    gain = 1.5;
    localEnhanced = gain * (homo_filter_out - localMean) + localMean;
    % Create a blurred version
    blurKernel = fspecial('gaussian', [15 15], 3);
    blurred = imfilter(localEnhanced, blurKernel, 'replicate');
    
    % Mask = Original - Blurred (this contains only the edges/details)
    mask = localEnhanced - blurred;
    
    % High-boost: Result = Original + k * Mask
    % If k=1, it's standard unsharp masking. If k > 1, it's high-boost.
    k = 2.0; 
    contrast_out = localEnhanced + k * mask;
    
    % Final clip and normalize to keep it in [0, 1]
    contrast_out = max(0, min(1, contrast_out));
end

function featureMap = generate_features(contrast_out)
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = [1 2 1; 0 0 0; -1 -2 -1];
    gradX = imfilter(contrast_out, Gx, 'replicate');
    gradY = imfilter(contrast_out, Gy, 'replicate');
    gradMag = sqrt(gradX.^2 + gradY.^2);
    if max(gradMag(:)) > 0
        gradMag = gradMag / max(gradMag(:));
    end

    hVar = ones(7,7) / (7*7);
    mu  = imfilter(contrast_out, hVar, 'replicate');
    mu2 = imfilter(contrast_out.^2, hVar, 'replicate');
    localVar = max(0, mu2 - mu.^2);
    if max(localVar(:)) > 0
        localVar = localVar / max(localVar(:));
    end

    % Go back to hypoechoic (dark tumor) assumption but with gentler smoothing
    intensityFeature = 1 - contrast_out;

    rawMap = (0.5 * intensityFeature) + (0.25 * gradMag) + (0.25 * localVar);

    % Single moderate gaussian — previous large sigma was destroying boundaries
    smoothedMap = imfilter(rawMap, fspecial('gaussian', [21 21], 4.0), 'replicate');

    featureMap = (smoothedMap - min(smoothedMap(:))) / ...
                 (max(smoothedMap(:)) - min(smoothedMap(:)));
end
function binaryMask = segment_image(featureMap)
    avgVal = mean(featureMap(:));
    stdVal = std(featureMap(:));

    % k=0.8 is a middle ground — strict enough to cut noise,
    % loose enough to capture a large lesion region
    k = 0.7;
    threshold = avgVal + (k * stdVal);
    binaryMask = featureMap > threshold;

    fprintf('Phase 5: Threshold set at %.4f (Mean: %.4f, Std: %.4f)\n', ...
            threshold, avgVal, stdVal);
end
function cleanMask = refine_mask(binaryMask)
    [rows, ~] = size(binaryMask);
    binaryMask(rows-10:rows, :) = 0;

    % FIX: Use smaller structuring element — radius 5 was eroding small
    % detections into nothing before they could be grown
    se_open  = strel('disk', 2);
    se_close = strel('disk', 8);  % larger closing to bridge internal gaps

    cleanMask = imopen(binaryMask, se_open);
    cleanMask = imfill(cleanMask, 'holes');
    cleanMask = imclose(cleanMask, se_close);
    cleanMask = imfill(cleanMask, 'holes');  % fill again after closing
end
%APPLY WATERSHEDDING TO FINAL MASK
function cleanMask = apply_watershed(cleanMask)
    % 1. Compute Distance Transform
    % Calculates the distance from every '1' to the nearest '0'.
    % The center of the tumor becomes the deepest point.
    D = bwdist(~cleanMask);
    
    % 2. Invert the distance map so the centers become 'Basins' (minima)
    D = -D;
    
    % 3. Force pixels outside the refinedMask to be 'Infinity'
    % This prevents the 'water' from leaking into the background.
    D(~cleanMask) = Inf;
    
    % 4. Apply Watershed
    % L contains labels for each 'flooded' basin.
    L = watershed(D);
    
    % 5. Extract the objects
    % Watershed lines (dams) are marked as 0. Everything else is a basin.
    cleanMask = (L > 0) & cleanMask;
    
    % 6. Optional: Close tiny gaps created by the watershed lines
    cleanMask = imclose(cleanMask, strel('disk', 1));
end

function finalMask = select_best_lesion(cleanMask)
    [L, num] = bwlabel(cleanMask);

    if num == 0
        finalMask = cleanMask;
        return;
    end

    stats = regionprops(L, 'Area', 'Centroid', 'Solidity');
    [rows, cols] = size(cleanMask);
    imgCenter = [cols/2, rows/2];

    scores = zeros(num, 1);
    for i = 1:num
        dist = sqrt(sum((stats(i).Centroid - imgCenter).^2));
        area = stats(i).Area;
        maxDist = sqrt((rows/2)^2 + (cols/2)^2);
        normDist = dist / maxDist;  % normalise to [0,1]

        % FIX: Area is the dominant signal — large solid regions are the lesion.
        % Distance penalty is mild and normalised so off-centre lesions aren't unfairly punished.
        scores(i) = (area * 1.0) + (stats(i).Solidity * 500) - (normDist * 200);
    end

    [~, bestIdx] = max(scores);
    finalMask = (L == bestIdx);
end


    