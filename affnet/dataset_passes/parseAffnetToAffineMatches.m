function [LAFS1,LAFS2,pts1,pts2] = parseAffnetToAffineMatches(matches_file)
    load(matches_file,'LAFS1','LAFS2','dists');
    pts1 = squeeze(LAFS1(:,1:2,3)+1);
    pts2 = squeeze(LAFS2(:,1:2,3)+1);
    LAFS1 = reshape(cell2mat(arrayfun(@(x) inv([squeeze(LAFS1(x,:,:)); 0 0 1]),1:size(LAFS1,1),'UniformOutput',false)),3,3,size(LAFS1,1));
    LAFS2 = reshape(cell2mat(arrayfun(@(x) inv([squeeze(LAFS2(x,:,:)); 0 0 1]),1:size(LAFS2,1),'UniformOutput',false)),3,3,size(LAFS2,1));
    
    [~,qualitySortedIdxs] = sort(dists,'Ascend');
    LAFS1 = LAFS1(:,:,qualitySortedIdxs);
    LAFS2 = LAFS2(:,:,qualitySortedIdxs);
    pts1 = pts1(qualitySortedIdxs,:);
    pts2 = pts2(qualitySortedIdxs,:);

end
