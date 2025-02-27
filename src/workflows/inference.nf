include {CSVCombiner as PatchCSVCombiner} from './modules/CSVHandlingUtils'
include {CSVCombiner as SlideCSVCombiner} from './modules/CSVHandlingUtils'

process SlideInference{
    label 'pytorch'
    input:
        path image_ids_path
        path model_path
    output:
        path '*results.csv'
    script:
    """
    python3 $projectDir/bin/SlideInference.py $image_ids_path $params.feature_dir_path $model_path $params.agg_type_label $params.scan_ds $params.class_order $params.meta_csv $params.results_file_name $params.results_path
    """
}
process PASInference{
    label 'pytorch'
    input:
        val image_ids
    output:
        path "Patch_Scores_*.csv"
        path "Slide_Scores_*.csv"
    script:
    """
    python3 $projectDir/bin/PASInference.py $image_ids $params.feature_dir_path $params.scan_ds
    """
}
process PatchInference{
    label 'pytorch'
    input:
        val image_ids
        path model_path
    output:
        path "Patch_Scores_*.csv"
        path "Mean_Patch_Scores_*.csv"
    script:
    """
    python3 $projectDir/bin/PatchInference.py $image_ids $params.feature_dir_path $model_path $params.scan_ds $params.class_order
    """
}

workflow infer_slide_scores {
    take:
        image_ids_path
        model_path
    main:
        SlideInference(image_ids_path,model_path)
}
workflow infer_patch_scores {
    take:
        image_id
        model_path
    main:
        PatchInference(image_id,model_path)
        PatchCSVCombiner(PatchInference.out[0].collect(),params.patch_scores_prefix)
        SlideCSVCombiner(PatchInference.out[1].collect(),params.slide_scores_prefix)
}
workflow infer_pas_scores {
    take:
        image_id
    main:
        PASInference(image_id)
        PatchCSVCombiner(PASInference.out[0].collect(),params.patch_scores_prefix)
        SlideCSVCombiner(PASInference.out[1].collect(),params.slide_scores_prefix)
}

//workflow{
//    Channel.fromPath(params.meta_csv)
//        .splitCSV(header: true, strip: true)
//        .map(row -> row.SlideID)
//        .set(image_id)
//    infer_scores(image_id)
//}