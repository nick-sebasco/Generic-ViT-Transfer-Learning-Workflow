include {vit_scanner; scan_aggregation} from './src/workflows/vitScanner.nf'
include {infer_scores} from './src/workflows/inference.nf'

workflow Inference {
    Channel.fromPath(params.meta_csv)
        .splitCsv(header: true, strip: true)
        .map{row -> row.SlideID.replaceAll(/[ -]/,"_")}
        .filter{ it != ""}
        .set{image_id}
    infer_scores(image_id)
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