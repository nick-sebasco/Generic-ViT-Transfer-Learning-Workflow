include {CSVCombiner as PatchCSVCombiner} from './modules/CSVHandlingUtils'
include {CSVCombiner as SlideCSVCombiner} from './modules/CSVHandlingUtils'

process Inference{
    label 'pytorch'
    input:
        val image_id
    output:
        path 'Patch_Scores_*.csv'
        path 'Slide_Scores_*.csv'
    script:
    """
    python3 $projectDir/bin/Inference.py $image_id $params.feature_dir_path $params.scan_ds
    """
}



workflow infer_scores {
    take:
        image_id
    main:
        Inference(image_id)
        PatchCSVCombiner(Inference.out[0].collect(),params.patch_scores_prefix)
        SlideCSVCombiner(Inference.out[1].collect(),params.slide_scores_prefix)
}

workflow{
    Channel.fromPath(params.meta_csv)
        .splitCSV(header: true, strip: true)
        .map(row -> row.SlideID)
        .set(image_id)
    infer_scores(image_id)
}