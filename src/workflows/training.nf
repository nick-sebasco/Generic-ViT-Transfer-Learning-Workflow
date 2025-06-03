include {CSVCombiner as PatchCSVCombiner} from './modules/CSVHandlingUtils'
include {CSVCombiner as SlideCSVCombiner} from './modules/CSVHandlingUtils'

//process TrainTestValSplit{
//    label 'pytorch'
//    input:
//        val image_id
//    output:
//        path 'Patch_Scores_*.csv'
//        path 'Slide_Scores_*.csv'
//    script:
//    """
//    python3 $projectDir/bin/Inference.py $image_id $params.feature_dir_path $params.scan_ds
//    """
//}

process TrainHead{
    label 'pytorch'
    input:
        path training_image_ids
        path validation_image_ids
    output:
        path 'models/*_final.pt'
    script:
    """
    python3 $projectDir/bin/run_training.py $training_image_ids $validation_image_ids $params.model_name $params.feature_dir_path $params.agg_type_label $params.scan_ds $params.meta_csv $params.target_column $params.class_order $params.ordinal --model_out_dir $params.model_dir_path --num_epochs $params.num_epochs
    """
}



workflow train_model {
    take:
        training_image_ids
        validation_image_ids
    main:
        TrainHead(training_image_ids,validation_image_ids)
    emit:
        TrainHead.out
}

