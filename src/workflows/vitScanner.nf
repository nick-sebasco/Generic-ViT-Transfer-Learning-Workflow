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

process scanAgg{
    label 'pytorch'
    input:
        val image_id
    output:
        val image_id
    script:
    """
    python3 $projectDir/bin/ScanAggregators.py $image_id  $params.feature_dir_path $params.scan_ds $params.agg_type $params.agg_window
    """
}

process qupath_to_zarr{
    label 'pytorch'
    input:
        path file_name
    
    script:
    """
    python3 $projectDir/bin/Qupath2ZarrMask.py $file_name  $params.qupath_mask_dir_path $params.roi_zarr_dir $params.qupath_ds $params.mask_names
    """
}

workflow qupath2zarr{
    take:
        file_name
    main:
        qupath_to_zarr(file_name)
}
workflow vit_scanner {
    take:
        image_id
    main:
        scanPrep(image_id)
        vitScan(scanPrep.out.transpose())
    emit:
    vitScan.out
}

workflow scan_aggregation {
    take: 
        image_id
    main:
        scanAgg(image_id)
    emit:
        scanAgg.out
}

workflow{
    Channel.fromPath(params.meta_csv)
        .splitCSV(header: true, strip: true)
        .map(row -> row.SlideID)
        .set(image_id)
    vit_scanner(image_id)
    scan_aggregation(vit_scanner.out.unique())
}