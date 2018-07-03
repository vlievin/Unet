from osgeo import gdal, osr,ogr
import numpy as np
import json

def vectorize_raster(geoJsonFileName,array2d,layerName="BuildingID",fieldName="BuildingID"):
    

    memdrv = gdal.GetDriverByName('MEM')
    src_ds = memdrv.Create('', array2d.shape[1], array2d.shape[0], 1)
    band = src_ds.GetRasterBand(1)
    band.WriteArray(array2d)

    dst_layername = "BuildingID"
    drv = ogr.GetDriverByName("geojson")
    dst_ds = drv.CreateDataSource(geoJsonFileName)
    dst_layer = dst_ds.CreateLayer(layerName, srs=None)

    fd = ogr.FieldDefn(fieldName, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 0

    gdal.Polygonize(band, None, dst_layer, dst_field, [], callback=None)

 
    
    return
def predict_score_batch(temporary_fold,batch_y,prediction):
       
    """
    Predicts scores considering IoU metrics following https://github.com/SpaceNetChallenge/utilities
    temporary_fold: path of the folder where to store temporary used geojson files extraced from raster input batch_y
    batch_y (np array): One hot pixel wise labels batch groundtruth(size_batch x WIDTH X HEIGHT x nb_channels) where nb_channels=2
               batch_y[:,:,:,0] should represent the background mask
               batch_y[:,:,:,1] should represent the buildings mask
    prediction (np array):  One hot pixel wise labels batch predidction (size_batch x WIDTH X HEIGHT x nb_channels) where nb_channels=2
    return averaged accuracy, f1 score, IoU of the batch
    
    """
    tot_score_batch=0
    tot_f1_score_batch=0
    tot_ious_batch=0

    for i in range(len(batch_y)):
        vectorize_raster(temporary_fold+str(i)+'_test_gt.geojson',batch_y[i])
        vectorize_raster(temporary_fold+str(i)+'_test_pred.geojson',prediction[i])
        with open(temporary_fold+str(i)+'_test_gt.geojson') as f:
            geojson_groundtruth = json.load(f)
        with open(temporary_fold+str(i)+'_test_pred.geojson') as f:
            geojson_prediction = json.load(f)
        
        
        M=len(geojson_prediction['features'])
        N=len(geojson_groundtruth['features'])
#         print('Image %d: %d predictions proposed and %d groundtruth'%(i,M,N))
        score=0
        IOUs_sum=0
        for feature_pred in geojson_prediction['features']:   
            IoUs=[]
            IoUs_accu=[]
#             print(ogr.CreateGeometryFromJson(json.dumps(feature_pred['geometry'])).GetArea())
#             print('Polygone')
            
            for feature_gt in geojson_groundtruth['features']:
#                 print(ogr.CreateGeometryFromJson(json.dumps(feature_gt['geometry'])).GetArea())
                poly1=ogr.CreateGeometryFromJson(json.dumps(feature_gt['geometry']))
                poly2=ogr.CreateGeometryFromJson(json.dumps(feature_pred['geometry']))
                intersection = poly1.Intersection(poly2)
                union = poly1.Union(poly2)
                if intersection is None:
                    IoUs.append(0.0)
                else:
                    IoUs.append(intersection.GetArea()/union.GetArea())
                
            IoUs=np.asarray(IoUs)
#             print(IoUs)
            IoUs_accu=(IoUs>0.5).astype(int)*IoUs
#             print(IoUs_accu)
            if (IoUs_accu.size and np.amax(IoUs_accu)>0):
                index=np.argmax(IoUs_accu)
#                 print('index %d'%index)
                geojson_groundtruth['features'].remove(geojson_groundtruth['features'][index])
#                 print('new size groundtruth %d'%len(geojson_groundtruth['features']))
                score+=1
                IOUs_sum+=IoUs[index]
#         print('score: %f: '%score)
#         print('IOUs_sum: %f: '%IOUs_sum)
        tot_ious_batch+=IOUs_sum/N
        tot_score_batch+=score/N
        tot_f1_score_batch+=2*score/(M+N)
    tot_ious_batch/=len(batch_y)
    tot_score_batch/=len(batch_y)
    tot_f1_score_batch/=len(batch_y)
    return tot_score_batch*100,tot_f1_score_batch*100,tot_ious_batch

        