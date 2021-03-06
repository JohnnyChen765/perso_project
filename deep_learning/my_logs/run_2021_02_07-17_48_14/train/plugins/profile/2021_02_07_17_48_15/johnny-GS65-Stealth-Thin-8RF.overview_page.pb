?	Pqx?|@Pqx?|@!Pqx?|@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Pqx?|@H?`๗ @1?^
?]??A?PlMK??I???l???*	?z?GyQ@2U
Iterator::Model::ParallelMapV2??72????!?}j?k!9@)??72????1?}j?k!9@:Preprocessing2F
Iterator::Model>???4`??!??)?I?F@)?7?k????1????'?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatˀ??,'??!???mH?7@)?3Lm????1.?m>}?3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6sHj?d??!??Բ9@)???????1ʥ?Nt#0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?Ɍ??^{?!??.r?#@)?Ɍ??^{?1??.r?#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\;Qi??!VH??K@)?4-?2j?1????>M@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?7k??*g?!?96?,/@)?7k??*g?1?96?,/@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?20.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI2????EV@Qn??(?%@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?`๗ @H?`๗ @!H?`๗ @      ??!       "	?^
?]???^
?]??!?^
?]??*      ??!       2	?PlMK???PlMK??!?PlMK??:	???l??????l???!???l???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q2????EV@yn??(?%@?"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsz:)I???!z:)I???"9
sequential_2/dense_6/MatMulMatMul? )?	ƶ?!?Gg?&??0"G
)gradient_tape/sequential_2/dense_7/MatMulMatMulϒj$w??!F??????0"9
sequential_2/dense_8/MatMulMatMul?????j??!??ߝj5??0"G
)gradient_tape/sequential_2/dense_6/MatMulMatMul???j??!???????0"G
)gradient_tape/sequential_2/dense_8/MatMulMatMul$.?آ?!J?????0"G
+gradient_tape/sequential_2/dense_8/MatMul_1MatMul$.?آ?!??Ek?x??"9
sequential_2/dense_8/SoftmaxSoftmax$.?آ?!?
???"G
+gradient_tape/sequential_2/dense_7/MatMul_1MatMul\??G}F??!????r~??"-
IteratorGetNext/_1_Send?'??h??!?:A??y??Q      Y@Y?|??9@a>????R@q.???j6@y4_??Ur??"?
both?Your program is POTENTIALLY input-bound because 67.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?20.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 