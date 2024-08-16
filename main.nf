include {vit_scanner} from './src/workflows/vitScanner.nf'
workflow{
    Channel.fromPath(params.meta_csv)
        .splitCsv(header: true, strip: true)
        .map{row -> row.SlideID.replaceAll(/[ -]/,"_")}
        .filter{ it != ""}
        .set{image_id}
    vit_scanner(image_id)
}