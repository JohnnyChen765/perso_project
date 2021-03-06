?	:?!?'@:?!?'@!:?!?'@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-:?!?'@????U?@1?I?>@A??ܵ??IcD?Ђ@*	
ףp=?S@2F
Iterator::Model?????!?a???D@)?	Q???1??d9?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????f??!?????9@)??_#I??1?#e?na4@:Preprocessing2U
Iterator::Model::ParallelMapV2y;?i????!1???3@)y;?i????11???3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz???3K??!"??9??6@)]??u???1?_?z\V)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW??:???!
?Krx?M@)6sHj?d??1??y??'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?2p@KW??!o*?%s$@)?2p@KW??1o*?%s$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor? :vp?!??I?ۙ@)? :vp?1??I?ۙ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?45.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIP	2?P@Q??_훃@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????U?@????U?@!????U?@      ??!       "	?I?>@?I?>@!?I?>@*      ??!       2	??ܵ????ܵ??!??ܵ??:	cD?Ђ@cD?Ђ@!cD?Ђ@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qP	2?P@y??_훃@@?"5
sequential/dense/MatMulMatMulF9?????!F9?????0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits???m.???!?Hۓ???"C
%gradient_tape/sequential/dense/MatMulMatMulxoN>_??!?n??U??0"-
IteratorGetNext/_4_Recv?/3????!?:)Q{???"3
Nadam/Nadam/update/add_2AddV2F9???v?!?a?q??"1
Nadam/Nadam/update/addAddV2?To??u?!n???????"9
Nadam/Nadam/update/truediv_3RealDiv=?-fJq?!>d?K???"3
Nadam/Nadam/update/add_1AddV2?
??5p?!?><????"1
Nadam/Nadam/update/mul_6Mul?
??5p?!?~???"9
Nadam/Nadam/update/truediv_2RealDiv?
??5p?!9??`???Q      Y@Y#%?T?/??ako??@?X@qY?\?V??@yl8?]<???"?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?45.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?31.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 