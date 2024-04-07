load('clean_dataset.mat');
load('25noise_dataset.mat');
vigilance=0.3;
M=20;
n=64;
%Patternslist = {patternA; patternB; patternC; patternD; patternE; patternF; patternG; patternH; patternI; patternJ; patternK; patternL; patternM; patternN; patternO; patternP; patternQ; patternR; patternS; patternT};
Patternslist = {noiseA_25; noiseB_25; noiseC_25; noiseD_25; noiseE_25; noiseF_25; noiseG_25; noiseH_25; noiseI_25; noiseJ_25; noiseK_25; noiseL_25; noiseM_25; noiseN_25; noiseO_25; noiseP_25; noiseQ_25; noiseR_25; noiseS_25; noiseT_25};
cleanPatterns = zeros(length(Patternslist), 64);
for item = 1:length(Patternslist)
    cleanPatterns(item, :) = reshape(Patternslist{item}, 1, []);
end
noiselist = {noiseA_25; noiseB_25; noiseC_25; noiseD_25; noiseE_25; noiseF_25; noiseG_25; noiseH_25; noiseI_25; noiseJ_25; noiseK_25; noiseL_25; noiseM_25; noiseN_25; noiseO_25; noiseP_25; noiseQ_25; noiseR_25; noiseS_25; noiseT_25};
noisePatterns = zeros(length(noiselist), 64);
for items = 1:length(noiselist)
    noisePatterns(items, :) = reshape(noiselist{items}, 1, []);
end

numPatterns = min(size(cleanPatterns, 1), size(cleanPatterns, 1));
overlapMatrix = zeros(numPatterns, numPatterns);

for i = 1:numPatterns
    for j = 1:numPatterns
        overlapMatrix(i, j) = sum(cleanPatterns(i, :) == cleanPatterns(j, :)) / numel(cleanPatterns(i, :));
    end
end
XTickLabels = arrayfun(@(x) char('A'+x-1), 1:numPatterns, 'UniformOutput', false);
YTickLabels = XTickLabels;
figure;
imagesc(overlapMatrix);
colorbar;
set(gca, 'XTick', 1:numPatterns, 'XTickLabel', XTickLabels);
set(gca, 'YTick', 1:numPatterns, 'YTickLabel', YTickLabels);
xlabel('Patterns');
ylabel('Patterns');
title('Confusion Matrix');
W = ones(M, n) * (1 / (1 + n));
V = ones(M, n);
categories = zeros(size(cleanPatterns, 1), 1);
categoryColumn = 2 * ones(1, M);

figure;
for i = 1:size(cleanPatterns, 1)
    X = cleanPatterns(i, :);
    subplot(M, M + 2, (M + 2) * (i - 1) + 1);
    imagesc(reshape(X, 8, 8));
    colormap(flipud(gray));
    title(['Input ', char(64+i)]);

    activeNodes = true(M, 1);
    while true

        Y = W * X';
        Y(~activeNodes) = -Inf;
        [value, J] = max(Y);
        normX = sum(X);
        S = sum(V(J, :) .* X) / normX;
        if S >= vigilance

            sumVX = sum(V(J, :) .* X);
            W(J, :) = (V(J, :) .* X) / (0.5 + sumVX);
            V(J, :) = X .* V(J, :);
            categories(i) = J;

    if categoryColumn(J) == 2
        categoryColumn(J) = max(categoryColumn) + 1;
    end
    subplot(M, M + 2, (M + 2) * (i - 1) + categoryColumn(J));
    %subplot(M, M + 1, (M + 1) * (i - 1) + categoryColumn(J));
    imagesc(reshape(V(J, :), 8, 8));
    colormap(flipud(gray));
    title(sprintf('Res %d', J));
    for previousCategory = 1:J-1
        if categoryColumn(previousCategory) > 2
            subplot(M, M + 2, (M + 2) * (i - 1) + categoryColumn(previousCategory));
            imagesc(reshape(V(previousCategory, :), 8, 8));
            colormap(flipud(gray));
        end
    end
            break;
        else
            activeNodes(J) = false;
            if ~any(activeNodes)
                activeNodes = true(M, 1);
                break;
            end
        end
    end 
    drawnow;
end
disp(categories)
VisualizeCategories(categories, M+1);



function VisualizeCategories(categories, M)
    figure;
    histogram(categories, 1:M);
    title('Histogram of Pattern Categories');
    xlabel('Category');
    ylabel('Number of Patterns');
    xlim([0 M]);
    grid on;
end
