matches_file = '/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/matches.mat';
I1 = imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img1.png');
I2 = imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img6.png');
[HstackSrc,HstackDst,srcPhaseZero, dstPhaseZero] = parseAffnetToAffineMatches(matches_file);

Hhats = reshape(cell2mat(arrayfun(@(x) squeeze(HstackSrc(:,:,x))\squeeze(HstackDst(:,:,x)),1:size(HstackSrc,3),'UniformOutput',false)),3,3,size(HstackSrc,3));
[ selectedIdxs ] = throwRedundantMatches( srcPhaseZero, dstPhaseZero, Hhats, 16);
HstackSrc =  HstackSrc(:,:,selectedIdxs);HstackDst =  HstackDst(:,:,selectedIdxs);
expandersList = expanderStuffParaDebug( HstackSrc, HstackDst,  rgb2gray(I1), rgb2gray(I2),true);
[ srcPtsTriplets, dstPtsTriplets, srcPtsCellAll,dstPtsCellAll,transformations] = expandersListToAffineMatchTriplets(expandersList);
matches.points.srcPts = cell2mat(srcPtsCellAll);
matches.points.dstPts = cell2mat(dstPtsCellAll);