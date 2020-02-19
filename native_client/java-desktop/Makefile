.PHONY: clean maven-install

include ../definitions.mk

clean:
	rm -rf jni/deepspeech_wrap.cpp
	rm -rf jni/build
	rm -rf libdeepspeech/src/main/java/org/mozilla/deepspeech/libdeepspeech/
	rm -rf libdeepspeech/src/main/resources/
	cd libdeepspeech ; mvn clean

jni/libs/libdeepspeech.so:
	-mkdir jni/libs
	cp ${TFDIR}/bazel-bin/native_client/libdeepspeech.so jni/libs/libdeepspeech.so

libdeepspeech/src/main/resources/libdeepspeech.so: 
	-mkdir libdeepspeech/src/main/resources
	cp ${TFDIR}/bazel-bin/native_client/libdeepspeech.so libdeepspeech/src/main/resources/libdeepspeech.so

jni/deepspeech_wrap.cpp: jni/deepspeech.i
	-mkdir -p libdeepspeech/src/main/java/org/mozilla/deepspeech/libdeepspeech/
	swig -c++ -java -package org.mozilla.deepspeech.libdeepspeech -outdir libdeepspeech/src/main/java/org/mozilla/deepspeech/libdeepspeech/ -o jni/deepspeech_wrap.cpp jni/deepspeech.i
	
jni/build/Makefile: jni/CMakeLists.txt
	cmake -S jni/ -B jni/build
	
jni/build/libdeepspeech-jni.so: jni/deepspeech_wrap.cpp jni/build/Makefile
	$(MAKE) -C jni/build

libdeepspeech/src/main/resources/libdeepspeech-jni.so: jni/build/libdeepspeech-jni.so
	-mkdir libdeepspeech/src/main/resources
	cp jni/build/libdeepspeech-jni.so libdeepspeech/src/main/resources/libdeepspeech-jni.so

maven-install: libdeepspeech/src/main/resources/libdeepspeech.so libdeepspeech/src/main/resources/libdeepspeech-jni.so
	cd libdeepspeech ; mvn install
