include {vit_scanner}
workflow{
    Channel.fromPath(params.meta_csv)
        .splitCSV(header: true, strip: true)
        .set(image_meta)
    vit_scanner(image_meta)
}