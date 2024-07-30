process scanPrep{
    label 'pytorch'
    input:
        val image_meta
    output:
        tuple val(image_meta), path('*_scan_group_*.csv')
    script:
    """
    python3 $projectDir/bin/ScanPrep.py $image_meta.ImageID --config $projectDir/$params.scan_params_config
    """
}
process vitScan{
    label 'pytorch'
    input:
        tuple val(image_meta), path(scan_group.csv)
    output:
        val image_meta
    script:
    """
    python3 $projectDir/bin/ViTScanner.py $image_meta.ImageID scan_group.csv --config $projectDir/$params.scan_params_config
    """
}

workflow vit_scanner {
    take:
        image_meta
    main:
        scanPrep(image_meta)
        vitScan(scanPrep.out.transpose())
}

workflow{
    Channel.fromPath(params.meta_csv)
        .splitCSV(header: true, strip: true)
        .set(image_meta)
    vit_scanner(image_meta)
}