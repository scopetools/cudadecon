REM ****OTFGEN****
radialft.exe .\2016_02_19_example_decon_deskew_data\mbPSF_560_NAp5nap42_z100nm.tif" .\2016_02_19_example_decon_deskew_data\mbOTF_560_NAp5nap42_z100nm.tif --nocleanup --fixorigin 10

REM ****DECONV****
cudaDeconv -z .36 -D 32.8 -R 32.8 -i 15 -M 0 0 1 -S --input-dir .\2016_02_19_example_decon_deskew_data\ERTKR --filename-pattern sample_scan_560 --otf-file .\2016_02_19_example_decon_deskew_data\mbOTF_560_NAp5nap42_z100nm.tif --NoBleachCorrection 
