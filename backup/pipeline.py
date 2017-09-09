import luigi
import config
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
import Image
import random
import gzip
import os
import struct
import re, shutil


import ast
class TupleParameter(luigi.Parameter):
    def parse(self, x):
        #print x
        return tuple(ast.literal_eval(str(x)))


class match_images(luigi.Task):
    dirs_1 = TupleParameter()
    files_1 = TupleParameter()
    def output(self):
        return luigi.LocalTarget('generated_match_images')
    def run(self):
        with self.output().open('w') as out_file:
        	index = 0
		while( index < len(self.files_1) - 1):
        		#print index
        		matches_counter = 1
        		person_images = []
        		person_images.append(self.files_1[index])
        		flag = True
        		while(flag):
                		next_index = index + matches_counter
                		#print next_index
                		if (next_index < len(self.files_1)) and (self.files_1[index][:4] == self.files_1[next_index][:4]):
                        		person_images.append(self.files_1[next_index])
                        		matches_counter+=1
                		else:
                        		flag = False
        		#print person_images
        		if(len(person_images) > 1):
                		split_count_test = 0
                		split_count_valid = 0
                		for i in range(0 , len(person_images)):
                        		img1 = plt.imread(os.path.join(config.input_dir , person_images[i]))
                        		img1 = (img1*255).round().astype(np.uint8)
                        		img1 = imresize(img1, (28 , 64))
                        		for j in range(i+1,len(person_images)):
                                		name = 'match' + str(person_images[i][:10]) + "_"+ str(person_images[j][:10]) + ".png"
                                		img2 = plt.imread(os.path.join(config.input_dir , person_images[j]))
                                		img2 = (img2*255).round().astype(np.uint8)
                                		img2 = imresize(img2 , (28,64))
                                		img = np.vstack((img1 , img2))
                                		img = Image.fromarray(img)
                                		if(split_count_valid < int(0.2*(len(person_images)))):
                                        		split_count_valid+=1
                                        		img.save(os.path.join(self.dirs_1[2],name))
                                		elif(split_count_test < int(0.2*(len(person_images)))):
                                        		split_count_test+=1
                                        		img.save(os.path.join(self.dirs_1[1],name))
                                		else:
                                        		img.save(os.path.join(self.dirs_1[0],name))


        		index = next_index

	    	out_file.write("Status : done")


class non_match_images(luigi.Task):
    dirs = TupleParameter()
    files = TupleParameter()
    def output(self):
        return luigi.LocalTarget('generated_non_match_images')
    def run(self):
        with self.output().open('w') as out_file:
            print "non matching"
            #validation_counter =0
	    #testing_counter =0
            for i in range(1, 1569):
                if (i < 10):
                    key = "000" + str(i)
                elif (i < 100):
                    key = "00" + str(i)
                elif (i < 1000):
                    key = "0" + str(i)
                else:
                    key = str(i)
                shortlisted_file_names = [filename for filename in self.files if filename[:4] == key]
                if shortlisted_file_names:
                    shortlisted = list(set(self.files) - set(shortlisted_file_names))
		    validation_counter = 0
		    testing_counter = 0 
                    for j in range(0, 54):
                        img1_key = random.choice(shortlisted_file_names)
                        img1 = plt.imread(os.path.join(config.input_dir, img1_key))
                        img1 = (img1 * 255).round().astype(np.uint8)
                        img1 = imresize(img1, (28, 64))
                        img2_key = random.choice(shortlisted)
                        shortlisted = list(set(shortlisted) - set(img2_key))
                        img2 = plt.imread(os.path.join(config.input_dir, img2_key))
                        name = "mis_match" + str(img1_key[:10]) + "_" + str(img2_key[:10]) + ".png"
                        img2 = (img2 * 255).round().astype(np.uint8)
                        img2 = imresize(img2, (28, 64))
                        img = np.vstack((img1, img2))
                        #img = img1 & img2
                        img = Image.fromarray(img)
                        if (validation_counter <5):
                            validation_counter += 1
                            img.save(os.path.join(self.dirs[2], name))
			elif(testing_counter < 5):
			    testing_counter += 1
			    img.save(os.path.join(self.dirs[1] , name))
                        else:
                            img.save(os.path.join(self.dirs[0], name))
            out_file.write("Status : done")


class training_data_generator(luigi.Task):
    def output(self):
        return luigi.LocalTarget('training_data_generator')
    def run(self):
        with self.output().open('w') as out_file:
            print "running training_data"
            files = [input_file for input_file in sorted(os.listdir(config.input_dir)) if '.png' in input_file]
            output_dir = os.getcwd()
            if not os.path.exists(config.training_dir):
                os.makedirs(config.training_dir)
            if not os.path.exists(config.testing_dir):
                os.makedirs(config.testing_dir)
	    if not os.path.exists(config.validation_dir):
		os.makedirs(config.validation_dir)
            training_dir = output_dir + '/' + config.training_dir + '/'
            testing_dir = output_dir + '/' + config.testing_dir + '/'
	    validation_dir = output_dir + '/' + config.validation_dir + '/'
            dirs = [training_dir, testing_dir, validation_dir]
            yield match_images(dirs , files)
            yield non_match_images(dirs, files)
            out_file.write("Status :done")



class encode(luigi.Task):
    input_dir = luigi.Parameter()
    images_filename = luigi.Parameter()
    labels_filename = luigi.Parameter()
    #perm = TupleParameter()
    def output(self):
        target_file = str(self.images_filename)+ str(self.labels_filename) +  "_encoding_status"
        return luigi.LocalTarget(target_file)

    def run(self):
        with self.output().open('w') as out_file:
	    #print "running encoding" , perm
	    #print "running encoding"
            if str(self.input_dir)[-1] != '/':
                self.input_dir += '/'
            l = os.listdir(str(self.input_dir))
            fs = [self.input_dir + x for x in sorted(l) if '.png' in x]
            self.num_imgs = len(fs)
            self.output_file_images = open(str(self.images_filename), "wb")
	    self.output_file_labels = open(str(self.labels_filename), "wb")
	    #########################Headers for image file#################
	    self.output_file_images.write(struct.pack('>i' , 2051)) #### Writes magic number for images
	    self.output_file_images.write(struct.pack('>i' , self.num_imgs))
	    im = np.asarray(Image.open(fs[0]).convert('L'), dtype=np.uint32)
            row , column = im.shape
            self.output_file_images.write(struct.pack('>i', row)) ####Writes number of rows
            self.output_file_images.write(struct.pack('>i', column)) #####Writes number of columns
	    ######################Headers for label file #############################
	    self.output_file_labels.write(struct.pack('>i' , 2049))
	    self.output_file_labels.write(struct.pack('>i' , self.num_imgs))
	
	    ################## Rest of the contents for the file ###################
	    for img in range(self.num_imgs):
			print img
			######### For images ######
			im = np.asarray(Image.open(fs[img]).convert('L'), dtype=np.uint32)
                	for i in xrange(im.shape[0]):
				for j in xrange(im.shape[1]):
					self.output_file_images.write(struct.pack('>B' , im[i,j]))
			##########For labels #####
			if re.match(self.input_dir + 'match.*', fs[img]):
                        	self.target = 1
			elif re.match(self.input_dir + 'mis_match.*', fs[img]):
                        	#print fs[img]
                        	self.target = 0
                	#print self.target
			self.target = np.uint32(self.target)
			self.output_file_labels.write(struct.pack('>B' , self.target))
	     #########################################################################3
	    self.output_file_images.close()
	    self.output_file_labels.close()
	    ### Images ##
	    f_in = open(str(self.images_filename))
	    f_out = gzip.open(str(self.images_filename) + '.gz' , 'wb')
            f_in_labels = open(str(self.labels_filename))
	    f_out_labels = gzip.open(str(self.labels_filename) + '.gz' , 'wb')
	    f_out.writelines(f_in)
	    f_out_labels.writelines(f_in_labels)
	    f_out.close()
	    f_out_labels.close()
	    f_in_labels.close()
            f_in.close()
	    os.remove(str(self.images_filename))
 	    os.remove(str(self.labels_filename))
	    out_file.write('Status : done')
            '''print (self.num_imgs)
	    if self.label == "label":
                self.magic_num = 2049
            else:
                self.magic_num = 2051
            self.output_file.write(struct.pack('>i', self.magic_num))
            self.output_file.write(struct.pack('>i', self.num_imgs))

            if self.label == "image":
                im = np.asarray(Image.open(fs[0]).convert('L'), dtype=np.uint32)
                r, c = im.shape
                self.output_file.write(struct.pack('>i', r))
                self.output_file.write(struct.pack('>i', c))

            for img in range(self.num_imgs):
                print img
                if self.label == "label":
                    if re.match(self.input_dir + 'match.*', fs[img]):
                        print fs[img]
                        self.target = 1
                    elif re.match(self.input_dir + 'mis_match.*', fs[img]):
                        print fs[img]
                        self.target = 0
                    #print self.target
                    self.target = np.uint32(self.target)
                    #print target
                    self.output_file.write(struct.pack('>B', self.target))
                else:
                    im = np.asarray(Image.open(fs[img]).convert('L'), dtype=np.uint32)
                    for i in xrange(im.shape[0]):
                        for j in xrange(im.shape[1]):
                            self.output_file.write(struct.pack('>B', im[i, j]))
            self.output_file.close()
            f_in = open(str(self.output_filename))
            f_out = gzip.open(str(self.output_filename) + '.gz', 'wb')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            os.remove(str(self.output_filename))
            #out_file.write('Status : done') 
            out_file.write('Status : done')
             '''
class encode_data(luigi.Task):
    def requires(self):
        return training_data_generator()

    def output(self):
        return luigi.LocalTarget('encode_data')

    def run(self):
        with self.output().open('w') as out_file:
            print "encoding_data"
            output_dir = os.getcwd()
            training_dir = output_dir + '/' + config.training_dir + '/'
            testing_dir = output_dir + '/' + config.testing_dir + '/'
	    validation_dir = output_dir + '/' + config.validation_dir + '/'
	    '''range_training = len(os.listdir(training_dir))
	    training_perm = np.random.permutation(range(range_training))
	    testing_perm = np.random.permutation(range(len(os.listdir(testing_dir))))
   	    validation_perm = np.random.permutation(range(len(os.listdir(validation_dir))))
	    training_files = sorted(os.listdir(str(training_dir)))
            training_files = [training_files[i] for i in training_perm]
	    testing_files = sorted(os.listdir(str(testing_dir)))
	    testing_files = [testing_files[j] for j in testing_perm]
	    validation_files = sorted(os.listdir(str(validation_dir)))
	    validation_files =[validation_files[k] for k in validation_perm]'''
            yield encode(training_dir, 'training-images-and-ubyte_20','training-labels-and-ubyte_20')
            yield encode(testing_dir, 'testing-images-and-ubyte_20','testing-labels-and-ubyte_20')
            #yield encode(testing_dir, 'testing-labels-and-ubyte_20',  label="label")
	    yield encode(validation_dir ,'validation-images-and-ubyte_20', 'validation-labels-and_ubyte_20')
	    #yield encode(validation_dir , 'validation-labels-and-ubyte_20', label='label')
            out_file.write("Status : done")



if __name__ == '__main__':
    luigi.run()
