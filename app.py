from flask import *
import os
from werkzeug import secure_filename
import pyocr
import pyocr.builders
import cv2
from PIL import Image
import numpy as np
from skimage.filters import threshold_local


def image_to_pdf(mode="pdf"):
	try:
		files_in_dir=os.listdir()
		#get image file names in current directory
		image_names=[]
		conventions=['jpeg','png','jpg']
		for file in files_in_dir:
			ext=file.split('.')[-1]
			if ext in conventions:
				image_names.append(file)

		curr_path=os.getcwd()

		#Read images into opencv numpy arrays
		images_read=[]
		for name in image_names:
			img=cv2.imread(name)
			images_read.insert(0,img)

		#Convert RGB images to Gray Scale
		thsh_images=[]
		for img in images_read:
			img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			clahe=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(16,16))
			img_gray=clahe.apply(img_gray)
			ret,th=cv2.threshold(img_gray,130,255,cv2.THRESH_BINARY)
			thsh_images.append(th)

		#Find contours in image using (tree retrival method) for hierarchy
		image_conts=[]
		for img in thsh_images:
			contours,_=cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			image_conts.append(contours)

		#Look for maximum area contour which describes page/rectangle structure in image
		max_area_conts=[]
		for contour in image_conts:
			max_ind,max_area=None,0
			for ind,cnt in enumerate(contour):
				area=cv2.contourArea(cnt)
				if area > max_area:
					max_area=area
					max_ind=ind
			max_area_conts.append(max_ind)

		#Draw closest four sided shape around maximum contour which is our 
		#area of interest in image
		approx_cont=[]
		for ind in range(len(images_read)):
			epsilon=0.02*cv2.arcLength(image_conts[ind][max_area_conts[ind]],True)
			approx=cv2.approxPolyDP(image_conts[ind][max_area_conts[ind]],epsilon,True)
			approx_cont.append(np.squeeze(approx))

		#Take out the four sided area of interest from image and
		#project to rectangle shape which is usual shape of an image.
		rect_images=[]
		for ind in range(len(images_read)):
			#top-left,bottom-left,bottom-right,top-right
			tl,bl,br,tr=approx_cont[ind].tolist()
			top_width=np.sqrt((tl[0]-tr[0])**2 + (tl[1]-tr[1])**2)
			bottom_width=np.sqrt((bl[0]-br[0])**2 + (bl[1]-br[1])**2)
			left_height=np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
			right_height=np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
			width=int(max(top_width,bottom_width))
			height=int(max(left_height,right_height))
			#order is tl,tr,br,bl
			pres=np.array([tl,tr,br,bl],dtype='float32')
			to=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype="float32")
			M=cv2.getPerspectiveTransform(pres,to)
			dst=cv2.warpPerspective(images_read[ind].copy(),M,(int(width),int(height)))
			rect_images.append(dst)

		#Digitise image in black and white as a scanned document
		digitised_image_names=[]
		for ind in range(len(rect_images)):
			img_gray=cv2.cvtColor(rect_images[ind].copy(),cv2.COLOR_BGR2GRAY)
			th=threshold_local(img_gray.copy(),101,offset=10,method="gaussian")
			img_gray=(img_gray>th)
			imgg=Image.fromarray(img_gray)
			size=(images_read[ind].shape[0],images_read[ind].shape[1])
			imgg.resize(size)
			name=curr_path+"/digitised_"+image_names[ind].split('.')[0]+'.jpg'
			digitised_image_names.append(name)
			imgg.save(digitised_image_names[ind])

		#Convert all digitised images to pdf format
		digitised_images=[]
		for name in digitised_image_names:
			imgg=Image.open(name)
			digitised_images.append(imgg)
		name=curr_path+"/digitised_images"+'.pdf'
		if len(digitised_images)>1:
			digitised_images[0].save(name,save_all=True,append_images=digitised_images[1:],resolution=100.0)
		else:
			digitised_images[0].save(name)

		if mode=="pdf":
			for file in digitised_image_names:
				os.remove(file)
			for file in image_names:
				os.remove(file)
			return

		elif mode=="text":
			#create text file
			name=curr_path+'/text'+'.txt'
			txt_file=open(name,"w")

			#Extract text from image using PyOcr
			tools=pyocr.get_available_tools()[0]
			lang=tools.get_available_languages()[0]
			for i,name in enumerate(digitised_image_names):
				txt=tools.image_to_string(Image.open(name), \
					lang=lang,builder=pyocr.builders.TextBuilder())
				txt=' '.join(txt.replace('-\n','').replace('\n',' ').split())
				txt_file.write("[Image "+str(i+1)+" text]\n\n")
				txt_file.write(txt)
				txt_file.write("\n\n")	

			txt_file.close()
			for file in digitised_image_names:
				os.remove(file)
			for file in image_names:
				os.remove(file)
			os.remove("digitised_images.pdf")
			return 

		elif mode=="speech":
			#Extract text from image using PyOcr
			image_txt=[]
			tools=pyocr.get_available_tools()[0]
			lang=tools.get_available_languages()[0]
			for name in digitised_image_names:
				txt=tools.image_to_string(Image.open(name), \
					lang=lang,builder=pyocr.builders.TextBuilder())
				txt=' '.join(txt.replace('-\n','').replace('\n',' ').split())
				image_txt.append(txt)

			for file in digitised_image_names:
				os.remove(file)
			for file in image_names:
				os.remove(file)
			os.remove("digitised_images.pdf")

			return image_txt

	except Exception:
		return


app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpeg','jpg','png']

@app.route('/uploadimages', methods=['POST','GET'])
def uploadimages():
	file_names=[]
	curr_path=os.getcwd()
	files_in_dir=os.listdir()
	for file in files_in_dir:
		if file[0]!='.' and file[0]!='_':
			if file not in ['static','templates','app.py','Procfile','requirements.txt']:
				os.remove(file)
	uploaded_files=request.files.getlist("files")
	for file in uploaded_files:
		if allowed_file(file.filename):
			file.filename=secure_filename(file.filename)
			file_names.insert(0,file.filename)
			file.save(file.filename)
	image_to_pdf(mode="pdf")
	try:
		return send_from_directory(os.getcwd(),'digitised_images.pdf',as_attachment=True)
	except Exception:
		abort(404)


@app.route('/uploadimage', methods=['POST'])
def uploadimage():
	try:
		file_names=[]
		curr_path=os.getcwd()
		files_in_dir=os.listdir()
		for file in files_in_dir:
			if file[0]!='.' and file[0]!='_':
				if file not in ['static','templates','app.py','Procfile','requirements.txt']:
					os.remove(file)
		uploaded_files=request.files.getlist("files")
		for file in uploaded_files:
			if allowed_file(file.filename):
				file.filename=secure_filename(file.filename)
				file_names.insert(0,file.filename)
				file.save(file.filename)
		image_to_pdf(mode="text")
		try:
			return send_from_directory(os.getcwd(),'text.txt',as_attachment=True)
		except Exception:
			abort(404)
	except Exception:
		return render_template("imagetotext.html")

@app.route('/uploadspeech', methods=['POST'])
def uploadspeech():
	try:
		file_names=[]
		curr_path=os.getcwd()
		files_in_dir=os.listdir()
		for file in files_in_dir:
			if file[0]!='.' and file[0]!='_':
				if file not in ['static','templates','app.py','Procfile','requirements.txt']:
					os.remove(file)
		uploaded_files=request.files.getlist("files")
		for file in uploaded_files:
			if allowed_file(file.filename):
				file.filename=secure_filename(file.filename)
				file_names.insert(0,file.filename)
				file.save(file.filename)
		txt=image_to_pdf(mode="speech")
		msg=False
		if len(txt)>0:
			msg=True
		try:
			return render_template('imagetospeech.html',msg=msg,txt=txt[0])
		except Exception:
			abort(404)
	except Exception:
		return render_template("imagetospeech.html")


@app.route('/')
@app.route('/index')
def home():
	return render_template('index.html')

@app.route('/imagetopdf')
def imagetopdf():
	return render_template('imagetopdf.html')

@app.route('/imagetotext')
def imagetotext():
	return render_template('imagetotext.html')

@app.route('/imagetospeech')
def imagetospeech():
	return render_template('imagetospeech.html')



if __name__=='__main__':
	app.run(debug=True)
