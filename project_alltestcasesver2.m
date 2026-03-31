function project()
    fprintf('======================================================\n');
    fprintf('STARTING TITAN PIPELINE: MAXIMUM IoU OVERRIDE\n');
    fprintf('======================================================\n\n');
    
    test_cases = {
        'TEST0.jpeg', 'TEST0GT.jpeg';
        'TEST1.jpeg', 'TEST1GT.jpeg';
        'TEST2.jpeg', 'TEST2GT.jpeg';
        'TEST3.jpeg', 'TEST3GT.jpeg'
    };
    
    num_cases = size(test_cases, 1);
    all_iou = -ones(num_cases, 1); 
    all_dice = -ones(num_cases, 1);
    
    fig_summary = figure('Name', 'Popup 1: All Test Cases (Final Results)', ...
                         'NumberTitle', 'off', 'Position', [50, 50, 1400, 950]);
    
    diagnostic_phases = struct();
    
    for i = 1:num_cases
        imgName = test_cases{i, 1};
        gtName  = test_cases{i, 2};
        
        fprintf('Processing Case %d/%d: [%s]...\n', i, num_cases, imgName);
        
        if ~exist(imgName, 'file') || ~exist(gtName, 'file')
            fprintf('  -> ERROR: Files %s or %s not found. Skipping.\n', imgName, gtName);
            continue;
        end
        
        [img, gt_resized, finalMask, metrics, phases] = process_single_image(imgName, gtName);
        
        all_iou(i) = metrics.iou;
        all_dice(i) = metrics.dice;
        
        fprintf('  -> IoU: %.4f | Dice: %.4f | Sens: %.4f | Spec: %.4f\n', ...
            metrics.iou, metrics.dice, metrics.sensitivity, metrics.specificity);
            
        if i == 1
            diagnostic_phases.img = img;
            diagnostic_phases.gt = gt_resized;
            diagnostic_phases.finalMask = finalMask;
            diagnostic_phases.data = phases;
            diagnostic_phases.name = imgName;
        end
        
        figure(fig_summary);
        
        subplot(num_cases, 3, (i-1)*3 + 1);
        imshow(img);
        if i == 1, title('Original Input', 'FontWeight', 'bold'); end
        ylabel(strrep(imgName, '.jpeg', ''), 'FontWeight', 'bold', 'Interpreter', 'none');
        
        subplot(num_cases, 3, (i-1)*3 + 2);
        imshow(gt_resized);
        if i == 1, title('Ground Truth Mask', 'FontWeight', 'bold'); end
        
        subplot(num_cases, 3, (i-1)*3 + 3);
        imshow(img);
        hold on;
        visboundaries(finalMask, 'Color', 'r', 'LineWidth', 2);
        visboundaries(gt_resized, 'Color', 'g', 'LineWidth', 1, 'LineStyle', '--');
        hold off;
        if i == 1, title('Red: Algorithm | Green: Truth', 'FontWeight', 'bold'); end
    end
    
    valid_cases = (all_iou >= 0); 
    mean_iou = mean(all_iou(valid_cases)); 
    mean_dice = mean(all_dice(valid_cases));
    
    fprintf('\n======================================================\n');
    fprintf('BATCH PROCESSING COMPLETE\n');
    fprintf('======================================================\n');
    fprintf('Total Images Processed: %d\n', sum(valid_cases));
    fprintf('MEAN IoU SCORE:         %.4f\n', mean_iou);
    fprintf('MEAN DICE SCORE:        %.4f\n', mean_dice);
    fprintf('======================================================\n');
    
    render_diagnostic_dashboard(diagnostic_phases);
end

% =========================================================================
% CORE PIPELINE (THE MAXIMUM IoU BRUTE FORCE METHOD)
% =========================================================================
function [img, gt_resized, finalMask, metrics, phases] = process_single_image(imgName, gtName)
    
    [img, gt] = preprocess_data(imgName, gtName);
    phases = struct(); 
    
    % Phase 1: Aggressive Denoising (Obliterates JPEG blocks and speckle)
    phases.img_denoised = imgaussfilt(medfilt2(img, [7 7], 'symmetric'), 1.5);
    
    % Phase 2: Contrast Enhancement
    phases.img_enhanced = adapthisteq(phases.img_denoised, 'ClipLimit', 0.02);
    
    % Phase 3: Multi-level K-Means Quantization (Finds the absolute darkest regions)
    levels = multithresh(phases.img_enhanced, 2); 
    quantized_img = imquantize(phases.img_enhanced, levels);
    phases.dark_mask = (quantized_img == 1);
    
    % Phase 4: Morphological "Glue" (Melts fractured tumor pieces back together)
    phases.glued_mask = imclose(phases.dark_mask, strel('disk', 12));
    phases.glued_mask = imfill(phases.glued_mask, 'holes');
    
    % Phase 5: Size Filtering (Deletes tiny noise specks)
    phases.clean_mask = bwareaopen(phases.glued_mask, 150);
    
    % Phase 6: Instant-Kill Heuristic Selection
    phases.best_candidate = select_ultimate_lesion(phases.clean_mask, phases.img_enhanced);
    
    % Phase 7: Halo-Expanding Active Contour
    % Negative bias (-0.15) forces the snake to expand and swallow the fuzzy gray boundaries!
    if sum(phases.best_candidate(:)) > 0
        finalMask = activecontour(phases.img_enhanced, phases.best_candidate, 100, 'Chan-Vese', ...
            'SmoothFactor', 1.5, 'ContractionBias', -0.15);
    else
        finalMask = false(size(img));
    end
    
    % Phase 8: Final Polish & Single Lesion Enforcer
    finalMask = imclose(finalMask, strel('disk', 5));
    finalMask = imfill(finalMask, 'holes');
    
    [L_final, num_final] = bwlabel(finalMask);
    if num_final > 1
        stats_final = regionprops(L_final, 'Area');
        [~, maxIdx] = max([stats_final.Area]);
        finalMask = (L_final == maxIdx);
    end
    
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
% HEURISTIC SELECTION (INSTANT KILL LOGIC)
% =========================================================================
function finalMask = select_ultimate_lesion(bw_mask, originalImg)
    [L, num] = bwlabel(bw_mask);
    if num == 0
        finalMask = bw_mask; 
        return; 
    end
    
    stats = regionprops(L, 'Area', 'Solidity', 'Centroid', 'BoundingBox', 'PixelIdxList');
    scores = zeros(num, 1);
    [rows, cols] = size(originalImg);
    centerX = cols / 2;
    centerY = rows / 2;
    maxDist = sqrt(centerX^2 + centerY^2);
    
    for i = 1:num
        bb = stats(i).BoundingBox;
        
        % INSTANT KILL 1: Acoustic Shadows (Touches bottom 15% of screen)
        if (bb(2) + bb(4)) > (rows * 0.85)
            continue; 
        end
        
        % INSTANT KILL 2: Skin/Gel Artifacts (Touches top 10% AND is wider than 40% of screen)
        if bb(2) < (rows * 0.10) && bb(3) > (cols * 0.40)
            continue; 
        end
        
        % Math Scoring
        areaScore = stats(i).Area; % Raw area heavily favored
        solidityScore = stats(i).Solidity;
        
        % Distance from center (Tumors are framed centrally by ultrasound techs)
        dist = sqrt((stats(i).Centroid(1) - centerX)^2 + (stats(i).Centroid(2) - centerY)^2);
        locationScore = 1 - (dist / maxDist);
        
        % Darkness Check
        meanInt = mean(originalImg(stats(i).PixelIdxList));
        contrastScore = 1 - meanInt;
        
        % The Multiplication ensures all factors must be high to win
        scores(i) = areaScore * locationScore * solidityScore * contrastScore;
    end
    
    [~, bestIdx] = max(scores);
    finalMask = (L == bestIdx);
end

% =========================================================================
% POPUP 2: SIMPLIFIED DIAGNOSTIC DASHBOARD RENDERER
% =========================================================================
function render_diagnostic_dashboard(dp)
    if isempty(fieldnames(dp)), return; end
    
    figure('Name', sprintf('Popup 2: Pipeline Phases (%s)', dp.name), ...
           'NumberTitle', 'off', 'Position', [100, 100, 1400, 600]);
           
    subplot(2, 4, 1); imshow(dp.img); title('1. Original Input');
    subplot(2, 4, 2); imshow(dp.data.img_denoised); title('2. Aggressive Denoise');
    subplot(2, 4, 3); imshow(dp.data.img_enhanced); title('3. Contrast Enhanced');
    subplot(2, 4, 4); imshow(dp.data.dark_mask); title('4. K-Means Quantization');
    
    subplot(2, 4, 5); imshow(dp.data.glued_mask); title('5. Morphological Glue');
    subplot(2, 4, 6); imshow(dp.data.clean_mask); title('6. Size Filtering');
    subplot(2, 4, 7); imshow(dp.data.best_candidate); title('7. Instant-Kill Selection');
    
    subplot(2, 4, 8); 
    imshow(dp.img); 
    title('8. Final Expanded Snake (Red)');
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
    
    % Clip extreme white/black dead pixels
    lower_bound = quantile(rawImg(:), 0.01);
    upper_bound = quantile(rawImg(:), 0.99);
    clipped = max(lower_bound, min(upper_bound, rawImg));
    imgNorm = (clipped - lower_bound) / (upper_bound - lower_bound);
    
    gtMask = rawGT > 0.5;
end