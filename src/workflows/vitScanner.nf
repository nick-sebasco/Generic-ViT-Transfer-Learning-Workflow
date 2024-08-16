process scanPrep{
    label 'pytorch'
    input:
        val image_id
    output:
        tuple val(image_id), path('*_scan_group_*.csv')
    script:
    """
    python3 $projectDir/bin/ScanPrep.py $image_id --config $projectDir/$params.scan_params_config
    """
}
process vitScan{
    label 'pytorch'
    input:
        tuple val(image_id), path("scan_group.csv")
    output:
        val image_id
    script:
    """
    python3 $projectDir/bin/ViTScanner.py $image_id scan_group.csv --config $projectDir/$params.scan_params_config
    """
}

workflow vit_scanner {
    take:
        image_id
    main:
        scanPrep(image_id)
        vitScan(scanPrep.out.transpose())
}

workflow{
    Channel.fromPath(params.meta_csv)
        .splitCSV(header: true, strip: true)
        .map(row -> row.SlideID)
        .set(image_id)
    vit_scanner(image_id)
}