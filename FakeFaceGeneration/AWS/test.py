
ACCESS_KEY='*****'
SECRET_KEY='*****'

import boto3


def compare_faces(sourceFile, targetFile):

    # client=boto3.client('rekognition')
    client=boto3.client('rekognition', region_name='us-east-1',
                        aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
                        
    # boto3.set_stream_logger('')
    imageSource=open(sourceFile,'rb')
    imageTarget=open(targetFile,'rb')

    response=client.compare_faces(SimilarityThreshold=80,
                                  SourceImage={'Bytes': imageSource.read()},
                                  TargetImage={'Bytes': imageTarget.read()})
    
    similarities=[]
    for faceMatch in response['FaceMatches']:
        position = faceMatch['Face']['BoundingBox']
        similarity = str(faceMatch['Similarity'])
        print('The face at ' +
               str(position['Left']) + ' ' +
               str(position['Top']) +
               ' matches with ' + similarity + '% confidence')
        similarities.append(similarity)

    imageSource.close()
    imageTarget.close()     
    # return len(response['FaceMatches'])          
    return response, similarities

from os.path import join as ospj
import glob
import pandas as pd
import numpy as np
import botocore

def main():
    # source_file='source'
    # target_file='target'

    # source_file=r'C:\GitHub\Smart-Education-data\001.png'
    # target_file=r'C:\GitHub\Smart-Education-data\003.png'


    n_src, n_ref=1,1

    #fawkes cloaked
    # source_dir=ospj(r'C:\GitHub\Smart-Education-data\data\celeba_sampled', f'src_s{n_src}_low_cloak_fawkes' )
    source_dir=r'C:\GitHub\Smart-Education-data\data\celeba_sampled\src_s1_low_cloak_jpg'
    target_dir=ospj(r'C:\GitHub\Smart-Education-data\data\celeba_sampled', f'ref_s{n_ref}')
    result_dir=r'C:\GitHub\Smart-Education-data\data\celeba_sampled\aws_result'
    result_f_name=f'fawkes_s{n_src}_similarity.csv'

    # source_dir=ospj(r'C:\GitHub\Smart-Education-data\data\celeba_sampled', f'src_s{n_src}_low_cloak_fawkes' )
    # target_dir=ospj(r'C:\GitHub\Smart-Education-data\data\celeba_sampled', f'ref_s{n_ref}')
    # result_dir=r'C:\GitHub\Smart-Education-data\data\celeba_sampled\aws_result'
    
    # list_source_files=glob.glob(ospj(source_dir, '*'))
    # list_target_files=glob.glob(ospj(target_dir, '*'))
    df_src_id_f_name=pd.read_csv(ospj(r'C:\GitHub\Smart-Education-data\data\celeba_cropped_178x178\comparison', f'src_s{n_src}_id_f_name.csv'), index_col=0)
    df_ref_id_f_name=pd.read_csv(ospj(r'C:\GitHub\Smart-Education-data\data\celeba_cropped_178x178\comparison', f'ref_s{n_ref}_id_f_name.csv'), index_col=0)
    
    # src_mat_f_names=df_src_id_f_name['f_name'].values.reshape(-1, n_src)
    # ref_mat_f_names=df_ref_id_f_name['f_name'].values.reshape(-1, n_ref)

    ### Warning: this supports n_src, n_ref = 1,1 only for now
    assert (n_src, n_ref) == (1, 1)

    #sampling
    N=100
    np.random.seed(0)
    # list_source_files=list(pd.Series(list_source_files, dtype='object').sample(n=N))
    # list_target_files=list(pd.Series(list_target_files, dtype='object').sample(n=N))
    
    df_src_id_f_name=df_src_id_f_name.sample(n=N)
    df_ref_id_f_name=df_ref_id_f_name.iloc[df_src_id_f_name.index]
    df_src_id_f_name.reset_index(drop=True)
    df_ref_id_f_name.reset_index(drop=True)
    # df_ref_id_f_name=df_ref_id_f_name.sample(n=N)

    result=[]
    for src_f_name, ref_f_name in zip(df_src_id_f_name['f_name'], df_ref_id_f_name['f_name']):
        source_file=ospj(source_dir, src_f_name)
        reference_file=ospj(target_dir, ref_f_name)
        # print(source_file)
        # print(reference_file)
        # break
        print(src_f_name, source_file)
        print(ref_f_name, reference_file)
        try:
            response, similarities=compare_faces(source_file, reference_file)
            result.append([src_f_name, ref_f_name, similarities])
        except:
            #languageaglksjglaksjgaldjla

    # result=[]
    # for source_file in list_source_files:
    #     for target_file in list_target_files:
    #         response, similarities=compare_faces(source_file, target_file)
    #         source_f_name=source_file.replace('\\', '/').split('/')[-1]
    #         target_f_name=target_file.replace('\\', '/').split('/')[-1]
    #         result.append([source_f_name, target_f_name, similarities])

    result=pd.DataFrame(result, columns=['source_f_name', 'target_f_name', 'similarities'])
    result.to_csv(ospj(result_dir, result_f_name))
    #when reading this file, use "df=pd.read_csv(<path>, index_col=0)"
    
    # face_matches=compare_faces(source_file, target_file)
    # print("Face matches: " + str(face_matches))

if __name__ == "__main__":
    main()