include {vit_scanner; scan_aggregation} from './src/workflows/vitScanner.nf'
include {infer_slide_scores; infer_patch_scores; infer_pas_scores} from './src/workflows/inference.nf'
include {train_model} from './src/workflows/training.nf'

workflow SlideInference {
    Channel.fromPath(params.test_image_ids_path)
        .set{image_ids}
    Channel.fromPath("${params.model_dir_path}/${params.model_name}_final.pt")
        .set{model}
    infer_slide_scores(image_ids,model)
}

workflow PatchInference {
    Channel.fromPath(params.meta_csv)
        .splitCsv(header: true, strip: true)
        .map{row -> row.SlideID.replaceAll(/[ -]/,"_")}
        .filter{ it != ""}
        .set{image_id}
    Channel.fromPath("${params.model_dir_path}/${params.model_name}_final.pt")
        .collect()
        .set{model}
    infer_patch_scores(image_id,model)
}

workflow PASInference {
    Channel.fromPath(params.meta_csv)
        .splitCsv(header: true, strip: true)
        .map{row -> row.SlideID.replaceAll(/[ -]/,"_")}
        .filter{ it != ""}
        .set{image_id}
    infer_pas_scores(image_id)
}

workflow Scan {
    Channel.fromPath(params.meta_csv)
        .splitCsv(header: true, strip: true)
        .map{row -> row.SlideID.replaceAll(/[ -]/,"_")}
        .filter{ it != ""}
        .set{image_id}
    vit_scanner(image_id)
}

workflow Agg {
    Channel.fromPath(params.meta_csv)
        .splitCsv(header: true, strip: true)
        .map{row -> row.SlideID.replaceAll(/[ -]/,"_")}
        .filter{ it != ""}
        .set{image_id}
    scan_aggregation(image_id)
}

workflow Scan_and_Agg{
    Channel.fromPath(params.meta_csv)
        .splitCsv(header: true, strip: true)
        .map{row -> row.SlideID.replaceAll(/[ -]/,"_")}
        .filter{ it != ""}
        .set{image_id}
    vit_scanner(image_id)
    scan_aggregation(vit_scanner.out.unique())
}

workflow Train{
    Channel.fromPath(params.training_image_ids_path).set{training_image_ids}
    Channel.fromPath(params.validation_image_ids_path).set{validation_image_ids}
    train_model(training_image_ids,validation_image_ids)
}
workflow TrainSlideInfer{
    Channel.fromPath(params.training_image_ids_path).set{training_image_ids}
    Channel.fromPath(params.validation_image_ids_path).set{validation_image_ids}
    Channel.fromPath(params.test_image_ids_path).set{test_image_ids}
    train_model(training_image_ids,validation_image_ids)
    infer_slide_scores(test_image_ids,train_model.out)
}