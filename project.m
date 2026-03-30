function project()
    
    imagePath = 'projectimg.png';
    maskPath  = 'GTmask.png';

    [img, gt] = preprocess_data(imagePath, maskPath);
    
    homo_filter_out = homomorphic_filter(img);
    wiener_out      = wiener_filter(homo_filter_out);
    adaptive_out    = adaptive_filter(wiener_out);
    contrast_out    = contrast_enhancement(adaptive_out);

    % FIX: Pass adaptive_out to generate_features instead of contrast_out.
    % adaptive_out still has the lesion as a clear dark blob because CLAHE
    % has not yet normalized the interior intensity away.
    % contrast_out is kept only for active_contour_refine which needs
    % sharp boundaries rather than a preserved dark blob.
    featureMap = generate_features(adaptive_out);
    
    binaryMask = segment_image(featureMap);
    cleanMask  = refine_mask(binaryMask);
    splitMask  = apply_watershed(cleanMask);
    roughMask  = select_best_lesion(splitMask);
    finalMask  = active_contour_refine(roughMask, contrast_out);

    figure('Name', 'Pipeline Debug', 'Position', [100 100 1800 900]);
    subplot(2,5,1);  imshow(img);            title('1. Raw Input');
    subplot(2,5,2);  imshow(homo_filter_out);title('2. Homomorphic');
    subplot(2,5,3);  imshow(wiener_out);     title('3. Wiener');
    subplot(2,5,4);  imshow(adaptive_out);   title('4. Adaptive');
    subplot(2,5,5);  imshow(contrast_out);   title('5. Contrast Enhanced');
    subplot(2,5,6);  imshow(featureMap);     title('6. Feature Map');
    subplot(2,5,7);  imshow(binaryMask);     title('7. Binary Mask');
    subplot(2,5,8);  imshow(cleanMask);      title('8. Refined Mask');
    subplot(2,5,9);  imshow(roughMask);      title('9. Rough Lesion');
    subplot(2,5,10); imshow(finalMask);      title('10. Final (Snake)');

    fprintf('--- Feature Map Stats ---\n');
    fprintf('Min: %.4f, Max: %.4f\n', min(featureMap(:)), max(featureMap(:)));
    fprintf('Mean: %.4f, Std: %.4f\n', mean(featureMap(:)), std(featureMap(:)));
    fprintf('Pixels above 0.5: %d\n', sum(featureMap(:) > 0.5));
    fprintf('Pixels above 0.7: %d\n', sum(featureMap(:) > 0.7));
    fprintf('Pixels above 0.9: %d\n', sum(featureMap(:) > 0.9));

    figure('Name', 'Overlay Comparison');
    overlayImg = zeros(size(img,1), size(img,2), 3);
    gt_resized = imresize(gt, [size(img,1), size(img,2)], 'nearest');
    overlayImg(:,:,1) = double(finalMask);
    overlayImg(:,:,2) = double(gt_resized);
    imshow(overlayImg);
    title('Red=finalMask | Green=GT | Yellow=Overlap');

    fprintf('Phase 1 Complete using %s and %s.\n', imagePath, maskPath);
end

% --- LOCAL FUNCTIONS ---

function [imgNorm, gtMask] = preprocess_data(imgP, mskP)
    rawImg = im2double(imread(imgP));
    rawGT  = im2double(imread(mskP));
    if size(rawImg, 3) == 3, rawImg = rgb2gray(rawImg); end
    if size(rawGT,  3) == 3, rawGT  = rgb2gray(rawGT);  end
    if any(size(rawGT) ~= size(rawImg))
        rawGT = imresize(rawGT, [size(rawImg,1), size(rawImg,2)], 'nearest');
        fprintf('Warning: GT mask resized to match image dimensions.\n');
    end
    lower   = quantile(rawImg(:), 0.01);
    upper   = quantile(rawImg(:), 0.99);
    clipped = max(lower, min(upper, rawImg));
    imgNorm = (clipped - lower) / (upper - lower);
    gtMask  = rawGT > 0.5;
end

function homo_filter_out = homomorphic_filter(img)
    imgLog  = log(1 + img);
    F       = fftshift(fft2(imgLog));
    [M, N]  = size(img);
    [U, V]  = meshgrid(1:N, 1:M);
    centerX = floor(N/2) + 1;
    centerY = floor(M/2) + 1;
    D       = sqrt((U - centerX).^2 + (V - centerY).^2);
    sigma   = 0.1 * mean([M, N]);
    gammaL  = 0.3;
    gammaH  = 1.8;
    H       = (gammaH - gammaL) * (1 - exp(-(D.^2) / (2 * sigma^2))) + gammaL;
    G       = F .* H;
    imgFiltered     = real(ifft2(ifftshift(G)));
    homo_filter_out = exp(imgFiltered) - 1;
    if max(homo_filter_out(:)) > min(homo_filter_out(:))
        homo_filter_out = (homo_filter_out - min(homo_filter_out(:))) / ...
                          (max(homo_filter_out(:)) - min(homo_filter_out(:)));
    end
end

function wiener_out = wiener_filter(img)
    wiener_out = wiener2(img, [5 5]);
    if max(wiener_out(:)) > min(wiener_out(:))
        wiener_out = (wiener_out - min(wiener_out(:))) / ...
                     (max(wiener_out(:)) - min(wiener_out(:)));
    end
    fprintf('Wiener filter complete\n');
end

function adaptive_out = adaptive_filter(img)
    [rows, cols] = size(img);
    adaptive_out = img;
    maxWin = 7;
    for r = 1:rows
        for c = 1:cols
            winSize = 3;
            done = false;
            while ~done && winSize <= maxWin
                rMin = max(1, r - floor(winSize/2));
                rMax = min(rows, r + floor(winSize/2));
                cMin = max(1, c - floor(winSize/2));
                cMax = min(cols, c + floor(winSize/2));
                window = img(rMin:rMax, cMin:cMax);
                zMin   = min(window(:));
                zMax   = max(window(:));
                zMed   = median(window(:));
                zxy    = img(r, c);
                if zMed > zMin && zMed < zMax
                    if zxy > zMin && zxy < zMax
                        adaptive_out(r, c) = zxy;
                    else
                        adaptive_out(r, c) = zMed;
                    end
                    done = true;
                else
                    winSize = winSize + 2;
                end
            end
            if ~done
                rMin = max(1, r - floor(maxWin/2));
                rMax = min(rows, r + floor(maxWin/2));
                cMin = max(1, c - floor(maxWin/2));
                cMax = min(cols, c + floor(maxWin/2));
                adaptive_out(r, c) = median(img(rMin:rMax, cMin:cMax), 'all');
            end
        end
    end
    if max(adaptive_out(:)) > min(adaptive_out(:))
        adaptive_out = (adaptive_out - min(adaptive_out(:))) / ...
                       (max(adaptive_out(:)) - min(adaptive_out(:)));
    end
    fprintf('Adaptive median filter complete\n');
end

function contrast_out = contrast_enhancement(homo_filter_out)
    [M, N] = size(homo_filter_out);
    tileSize    = max(2, round(min(M,N) / 8));
    numTilesRow = max(2, round(M / tileSize));
    numTilesCol = max(2, round(N / tileSize));
    noiseEstimate = std(homo_filter_out(:));
    clipLimit = max(0.005, min(0.04, 0.02 / noiseEstimate));
    contrast_out = adapthisteq(homo_filter_out, ...
        'ClipLimit',    clipLimit, ...
        'NumTiles',     [numTilesRow numTilesCol], ...
        'Distribution', 'rayleigh');
    contrast_out = anisodiff2D(contrast_out, 3, 0.15, 30, 1);
end

function imgOut = anisodiff2D(img, niter, lambda, kappa, option)
    imgOut = img;
    for i = 1:niter
        deltaN = circshift(imgOut, [-1  0]) - imgOut;
        deltaS = circshift(imgOut, [ 1  0]) - imgOut;
        deltaE = circshift(imgOut, [ 0  1]) - imgOut;
        deltaW = circshift(imgOut, [ 0 -1]) - imgOut;
        if option == 1
            cN = exp(-(deltaN/kappa).^2);
            cS = exp(-(deltaS/kappa).^2);
            cE = exp(-(deltaE/kappa).^2);
            cW = exp(-(deltaW/kappa).^2);
        else
            cN = 1 ./ (1 + (deltaN/kappa).^2);
            cS = 1 ./ (1 + (deltaS/kappa).^2);
            cE = 1 ./ (1 + (deltaE/kappa).^2);
            cW = 1 ./ (1 + (deltaW/kappa).^2);
        end
        imgOut = imgOut + lambda * (cN.*deltaN + cS.*deltaS + ...
                                    cE.*deltaE + cW.*deltaW);
    end
end

function featureMap = generate_features(adaptive_out)
    % Uses adaptive_out directly — CLAHE has not been applied yet so
    % the lesion is still a clear dark blob with simple intensity contrast.
    % Multi-scale dark detection will correctly produce a bright response
    % over the lesion interior.
    bg_small = imfilter(adaptive_out, fspecial('gaussian', [21 21],  5), 'replicate');
    bg_med   = imfilter(adaptive_out, fspecial('gaussian', [41 41], 10), 'replicate');
    bg_large = imfilter(adaptive_out, fspecial('gaussian', [61 61], 15), 'replicate');

    dark_small = max(0, bg_small - adaptive_out);
    dark_med   = max(0, bg_med   - adaptive_out);
    dark_large = max(0, bg_large - adaptive_out);

    if max(dark_small(:)) > 0, dark_small = dark_small / max(dark_small(:)); end
    if max(dark_med(:))   > 0, dark_med   = dark_med   / max(dark_med(:));   end
    if max(dark_large(:)) > 0, dark_large = dark_large / max(dark_large(:)); end

    rawMap     = (0.2 * dark_small) + (0.3 * dark_med) + (0.5 * dark_large);
    featureMap = imfilter(rawMap, fspecial('gaussian', [31 31], 8), 'replicate');
    featureMap = (featureMap - min(featureMap(:))) / ...
                 (max(featureMap(:)) - min(featureMap(:)));

    fprintf('Feature map — min: %.4f max: %.4f mean: %.4f\n', ...
        min(featureMap(:)), max(featureMap(:)), mean(featureMap(:)));
end

function binaryMask = segment_image(featureMap)
    avgVal    = mean(featureMap(:));
    stdVal    = std(featureMap(:));
    k         = 0.75;
    threshold = avgVal + (k * stdVal);
    binaryMask = featureMap > threshold;
    fprintf('Threshold: %.4f (mean: %.4f std: %.4f)\n', threshold, avgVal, stdVal);
end

function cleanMask = refine_mask(binaryMask)
    [rows, ~] = size(binaryMask);
    binaryMask(rows-10:rows, :) = 0;

    se_open = strel('disk', 2);
    cleanMask = imopen(binaryMask, se_open);

    se_seal = strel('disk', 12);
    cleanMask = imclose(cleanMask, se_seal);
    cleanMask = imfill(cleanMask, 'holes');

    se_smooth = strel('disk', 4);
    cleanMask = imclose(cleanMask, se_smooth);
    cleanMask = imfill(cleanMask, 'holes');
end

function cleanMask = apply_watershed(cleanMask)
    D             = bwdist(~cleanMask);
    D             = -D;
    D(~cleanMask) = Inf;
    L             = watershed(D);
    cleanMask     = (L > 0) & cleanMask;
    cleanMask     = imclose(cleanMask, strel('disk', 1));
end

function finalMask = select_best_lesion(cleanMask)
    [L, num] = bwlabel(cleanMask);
    if num == 0
        finalMask = cleanMask;
        return;
    end
    stats        = regionprops(L, 'Area', 'Centroid', 'Solidity', 'BoundingBox');
    [rows, cols] = size(cleanMask);
    imgCenter    = [cols/2, rows/2];
    maxDist      = sqrt((rows/2)^2 + (cols/2)^2);
    maxArea      = max([stats.Area]);
    edgeMargin   = round(min(rows, cols) * 0.05);
    scores       = zeros(num, 1);
    for i = 1:num
        dist     = sqrt(sum((stats(i).Centroid - imgCenter).^2));
        normDist = dist / maxDist;
        normArea = stats(i).Area / maxArea;
        if normArea < 0.05
            scores(i) = -Inf;
            continue;
        end
        bb   = stats(i).BoundingBox;
        xMin = bb(1); yMin = bb(2);
        xMax = bb(1) + bb(3); yMax = bb(2) + bb(4);
        touchesEdge = (xMin <= edgeMargin) || ...
                      (yMin <= edgeMargin) || ...
                      (xMax >= cols - edgeMargin) || ...
                      (yMax >= rows - edgeMargin);
        edgePenalty = 0;
        if touchesEdge
            edgePenalty = 200;
        end
        scores(i) = (normArea * 500) + (stats(i).Solidity * 300) ...
                    - (normDist * 100) - edgePenalty;
    end
    [~, bestIdx] = max(scores);
    finalMask    = (L == bestIdx);
end

function finalMask = active_contour_refine(roughMask, contrast_out)
    se_shrink = strel('disk', 10);
    initMask  = imerode(roughMask, se_shrink);
    if sum(initMask(:)) == 0
        initMask = imerode(roughMask, strel('disk', 3));
    end
    if sum(initMask(:)) == 0
        finalMask = roughMask;
        fprintf('Active contour skipped — initial mask too small after erosion\n');
        return;
    end
    finalMask = activecontour(contrast_out, initMask, 200, 'Chan-Vese', ...
        'SmoothFactor',    3, ...
        'ContractionBias', -0.05);
    fprintf('Active contour refinement complete\n');
end