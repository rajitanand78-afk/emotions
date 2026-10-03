import cv2
from deepface import DeepFace
C={"happy":(0,180,0),"sad":(0,0,220),"neutral":(0,220,220)}
cam=cv2.VideoCapture(0)
det=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
e="neutral"
n=0
while True:
    ok,f=cam.read()
    if not ok: break
    n+=1
    faces=det.detectMultiScale(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(60,60))
    if len(faces):
        x,y,w,h=max(faces,key=lambda b:b[2]*b[3])
        cv2.rectangle(f,(x,y),(x+w,y+h),(255,255,255),2)
        if n%8==0:
            r=DeepFace.analyze(f[y:y+h,x:x+w],actions=["emotion"],detector_backend="skip",enforce_detection=False,silent=True)
            r=r[0] if isinstance(r,list) else r
            e=max(C,key=lambda x:r["emotion"][x])
            s=f"{e.capitalize()} ({r['emotion'][e]:.0f}%)"
        else:s=e.capitalize()
    else:s="No face detected"
    t=f.copy()
    t[:]=C[e]
    f=cv2.addWeighted(f,.78,t,.22,0)
    cv2.putText(f,"Emotion: "+s,(20,35),0,.8,(255,255,255),2)
    cv2.imshow("Emotion Camera",f)
    if cv2.waitKey(1)&255==ord("q"):break
cam.release()
cv2.destroyAllWindows()