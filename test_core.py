import io
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from cellcount.inference import (Detection, letterbox, decode_yolo, classwise_nms,
                                 as_counts, read_image, quality_check)
from cellcount.classical import detect_classical
from cellcount.artifacts import import_colab_bundle
from cellcount.workflow import process_image, make_summary


def test_letterbox_rectangular_bccd_photo():
    img = np.zeros((480,640,3),dtype=np.uint8)
    blob, ratio, left, top = letterbox(img,640)
    assert blob.shape == (1,3,640,640)
    assert ratio == 1.0 and left == 0 and top == 80
    assert np.all(blob[0,:,0,0] == 114 / 255.0)


def test_raw_yolo_decoding_and_reverse_letterbox():
    # One synthetic ONNX tensor: [cx,cy,w,h, class_RBC, class_WBC, class_platelet]
    # BCCD original is 640 x 480, so y has 80 pixel padded top offset.
    x = np.zeros((1,7,2),dtype=np.float32)
    x[0,:,0] = [100, 180, 40, 20, .92, .08, .03]
    x[0,:,1] = [300, 280, 20, 20, .02, .04, .91]
    dets = decode_yolo(x, ratio=1, left=0, top=80, original_shape=(480,640))
    assert len(dets) == 2
    assert as_counts(dets) == {'RBC': 1, 'WBC': 0, 'Platelets': 1}
    rbc = next(d for d in dets if d.class_id == 0)
    assert np.allclose(rbc.box, (80, 90, 120, 110))


def test_nms_suppresses_same_class_and_keeps_other_class():
    boxes=np.array([[0,0,10,10],[1,1,11,11],[1,1,11,11]],dtype=float)
    scores=np.array([.9,.8,.7])
    labels=np.array([0,0,2])
    ids=classwise_nms(boxes,scores,labels,iou_threshold=.5)
    assert set(ids)=={0,2}


def test_invalid_image_rejected():
    with pytest.raises(ValueError,match='Unsupported'):
        read_image(b'this is not an image')


def test_classical_no_fake_confidence_and_review_export():
    img=np.full((120,160,3),235,dtype=np.uint8)
    cv2.circle(img,(70,60),15,(150,140,190),-1)
    ok, encoded=cv2.imencode('.png',img)
    assert ok
    r=process_image(encoded.tobytes(),'Classical baseline')
    assert set(r['counts'])=={'RBC','WBC','Platelets'}
    assert r['method']=='Classical baseline'
    assert r['mean_confidence'] is None
    assert r['quality']['status'] in ['REVIEW REQUIRED','NEEDS MANUAL COUNT']
    review = {r['id']: {'RBC': 2,'WBC': 1,'Platelets': 0}}
    approvals={r['id']:True}
    export=make_summary([r],review,approvals)
    assert export.loc[0,'RBC_manual']==2
    assert export.loc[0,'technician_confirmed']
    assert export.loc[0,'sample_id']=='SAMPLE_001'
    assert 'filename' not in export.columns


def test_import_only_allowlisted_artifacts(tmp_path):
    stream=io.BytesIO()
    with ZipFile(stream,'w') as archive:
        archive.writestr('artifacts/yolo_test_overall.json','{"mAP50":0.42}')
        archive.writestr('artifacts/cellcount_yolo11n.onnx',b'fake model (test only)')
        archive.writestr('../secret.txt',b'bad')
        archive.writestr('data/patient_photo.png',b'private')
    extracted=import_colab_bundle(stream.getvalue(),tmp_path)
    assert set(extracted)=={'cellcount_yolo11n.onnx','yolo_test_overall.json'}
    assert not (tmp_path/'secret.txt').exists()
    assert not (tmp_path/'patient_photo.png').exists()


def test_raw_empty_predictions():
    zero = np.zeros((1,7,0),dtype=np.float32)
    assert decode_yolo(zero,1.0,0,0,(480,640))==[]


def test_onnx_shape_wrong_fails_loudly():
    with pytest.raises(ValueError,match='Unsupported YOLO output'):
        decode_yolo(np.zeros((1,9,10),dtype=np.float32),1,0,0,(480,640))


def test_paired_sr_uses_same_low_resolution_image_for_control_and_enhancement():
    import numpy as np
    class FakeDetector:
        def __init__(self): self.shapes = []
        def detect(self, bgr, confidence, iou):
            self.shapes.append(bgr.shape[:2])
            return [Detection(0,0.6,(0,0,10,10))], 0.01
    class FakeUpscaler:
        def enhance(self, bgr):
            assert bgr.shape[:2] == (120,160)
            return np.repeat(np.repeat(bgr,4,axis=0),4,axis=1), 0.05
    img=np.full((480,640,3),235,dtype=np.uint8)
    cv2.circle(img,(70,60),15,(150,140,190),-1)
    ok, encoded=cv2.imencode('.png',img)
    assert ok
    detector=FakeDetector()
    result=process_image(encoded.tobytes(),'YOLO11n',detector=detector,
                         enhancer=FakeUpscaler(),compare_sr=True)
    assert detector.shapes == [(480,640),(120,160),(480,640)]
    assert result['sr']['raw_low_res_counts']['RBC']==1
    assert result['sr']['counts']['RBC']==1
    assert result['sr']['raw_low_res_seconds']==0.01
