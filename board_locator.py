"""Locate a visible, axis-aligned two-colour 8x8 board; no screen coordinates stored."""
from dataclasses import dataclass
from collections import Counter
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


@dataclass(frozen=True)
class BoardRegion:
    x: int
    y: int
    size: int
    confidence: float = 0.


class BoardLocator:
    @staticmethod
    def confidence(image, region=None):
        if region is None:
            h,w=image.shape[:2]
            if abs(h-w)>3: return 0.
            region=BoardRegion(0,0,min(h,w))
        x,y,s=region.x,region.y,region.size
        if s<128 or x<0 or y<0 or x+s>image.shape[1] or y+s>image.shape[0]: return 0.
        offsets=np.array([.12,.88])
        cells=np.arange(8)
        xx=x+(cells[:,None]+offsets)*s/8
        yy=y+(cells[:,None]+offsets)*s/8
        # Four corners per square: pieces, move highlights and coordinate labels
        # are tolerated, but an obscuring window must not be interpreted as chess.
        samples=image[yy.astype(int)[:,None,:,None],xx.astype(int)[None,:,None,:],:3].astype(float)
        tiles=np.median(samples,axis=(2,3))
        parity=np.indices((8,8)).sum(axis=0)%2
        colors=np.stack([np.median(tiles[parity==i],axis=0) for i in (0,1)])
        if np.max(np.abs(colors[0]-colors[1]))<28: return 0.
        errors=np.max(np.abs(tiles-colors[parity]),axis=2)
        return float(np.mean(errors<24))

    @staticmethod
    def refine(image, region):
        # Fit the seven internal grid boundaries at native resolution. Resizing
        # the search image otherwise introduces a few pixels of cumulative drift.
        x,y,size=region.x,region.y,region.size
        pad=max(5,round(size*.02))
        crop=image[max(0,y+pad):min(image.shape[0],y+size-pad),
                   max(0,x-pad):min(image.shape[1],x+size+pad),:3].astype(float)
        energy_x=np.median(np.max(np.abs(np.diff(crop,axis=1)),axis=2),axis=0)
        crop=image[max(0,y-pad):min(image.shape[0],y+size+pad),
                   max(0,x+pad):min(image.shape[1],x+size-pad),:3].astype(float)
        energy_y=np.median(np.max(np.abs(np.diff(crop,axis=0)),axis=2),axis=1)
        def fit(energy, origin):
            base=max(0,origin-pad)
            best=(-1,origin,size)
            for start in range(max(0,origin-pad),origin+pad+1):
                for side in range(max(128,size-pad),size+pad+1):
                    indices=np.round(start+np.arange(1,8)*side/8).astype(int)-base-1
                    if indices.min()<0 or indices.max()>=len(energy): continue
                    score=float(energy[indices].mean())
                    if score>best[0]: best=(score,start,side)
            return best[1:]
        xx,sx=fit(energy_x,x)
        yy,sy=fit(energy_y,y)
        refined=BoardRegion(xx,yy,round((sx+sy)/2))
        confidence=BoardLocator.confidence(image,refined)
        return BoardRegion(refined.x,refined.y,refined.size,confidence) if confidence>=.88 else region

    def find(self, image, previous=None):
        if previous is not None:
            confidence=self.confidence(image,previous)
            if confidence>=.94:
                return BoardRegion(previous.x,previous.y,previous.size,confidence)
        scale=min(1.,1200/image.shape[1])
        small=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else image
        h,w=small.shape[:2]
        pixels=small[::3,::3,:3].reshape(-1,3)
        bins=(pixels.astype(np.int32)//24)
        codes=bins[:,0]*121+bins[:,1]*11+bins[:,2]
        palette=Counter(codes.tolist()).most_common(16)
        rectangles=set()
        for code,count in palette:
            if count<40: continue
            color=np.median(pixels[codes==code],axis=0)
            mask=(np.max(np.abs(small[:,:,:3].astype(np.int16)-color),axis=2)<17).astype(np.uint8)
            n,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=4)
            for x,y,tw,th,area in stats[1:]:
                if 16<=tw<=min(w,h)/8+3 and abs(tw-th)<=2 and area>tw*th*.35:
                    rectangles.add((int(x),int(y),int(round((tw+th)/2))))
        # Group square sizes before voting for a complete 8x8 origin.
        sizes=Counter(s for _,_,s in rectangles)
        proposals=[]
        for size,_ in sizes.most_common(12):
            tiles=[r for r in rectangles if abs(r[2]-size)<=1]
            if len(tiles)<10: continue
            size=float(np.median([r[2] for r in tiles]))
            # Colour masks can lose one antialiased border pixel. Grid pitch is
            # the distance between tile edges, rather than the mask's width.
            steps=[]
            for axis in (0,1):
                coordinates=sorted(set(r[axis] for r in tiles))
                steps.extend(b-a for a,b in zip(coordinates,coordinates[1:]) if abs((b-a)-size)<=3)
            if len(steps)>=4:
                size=float(np.median(steps))
            votes=Counter()
            for x,y,_ in tiles:
                for col in range(8):
                    left=x-col*size
                    if left < -1 or left+8*size>w+1: continue
                    for row in range(8):
                        top=y-row*size
                        if top < -1 or top+8*size>h+1: continue
                        votes[(round(left/3),round(top/3))]+=1
            for (vx,vy),count in votes.most_common(10):
                if count<10: continue
                # Refine voting bins using the observed square edges.
                lefts=[x-round((x-vx*3)/size)*size for x,_,_ in tiles]
                tops=[y-round((y-vy*3)/size)*size for _,y,_ in tiles]
                left=round(float(np.median([v for v in lefts if abs(v-vx*3)<3.1])))
                top=round(float(np.median([v for v in tops if abs(v-vy*3)<3.1])))
                region=BoardRegion(max(0,left),max(0,top),round(8*size))
                score=self.confidence(small,region)
                if score>=.88:
                    actual=BoardRegion(round(region.x/scale),round(region.y/scale),round(region.size/scale),score)
                    distance=0 if previous is None else abs(actual.x-previous.x)+abs(actual.y-previous.y)+abs(actual.size-previous.size)
                    proposals.append((score,-distance,actual))
        return self.refine(image,max(proposals,key=lambda p:p[:2])[2]) if proposals else None


class DetectionWorker(QThread):
    found=Signal(object, int)
    failed=Signal(str, int)

    def __init__(self,image,previous,generation,parent=None):
        super().__init__(parent)
        self.image=image
        self.previous=previous
        self.generation=generation

    def run(self):
        try:
            self.found.emit(BoardLocator().find(self.image,self.previous),self.generation)
        except Exception as exc:
            self.failed.emit(str(exc),self.generation)
