import 'package:hard_diffusion/forms/advanced_column.dart';
import 'package:hard_diffusion/forms/prompt_column.dart';
import 'package:html/parser.dart' show parse;
import 'package:flutter/material.dart';
import 'package:hard_diffusion/api/network_service.dart';
import 'package:hard_diffusion/vertical_separator.dart';

class InferTextToImageForm extends StatefulWidget {
  const InferTextToImageForm({super.key});

  @override
  InferTextToImageFormState createState() {
    return InferTextToImageFormState();
  }
}

class InferTextToImageFormState extends State<InferTextToImageForm> {
  final _formKey = GlobalKey<FormState>();
  String prompt = "";

  void setPrompt(value) {
    prompt = value;
  }

  String negativePrompt = "";

  void setNegativePrompt(value) {
    negativePrompt = value;
  }

  bool useMultipleModels = false;

  void setUseMultipleModels(value) {
    useMultipleModels = value;
  }

  bool useNsfw = false;

  void setUseNsfw(value) {
    useNsfw = value;
  }

  bool useAdvanced = true;

  void setUseAdvanced(value) {
    setState(() {
      useAdvanced = value;
    });
  }

  bool useRandomSeed = false;

  void setUseRandomSeed(value) {
    useRandomSeed = value;
  }

  int seed = 0;

  void setSeed(value) {
    seed = value;
  }

  int width = 512;

  void setWidth(value) {
    width = value;
  }

  int height = 512;

  void setHeight(value) {
    height = value;
  }

  int inferenceSteps = 50;

  void setInferenceSteps(value) {
    inferenceSteps = value;
  }

  double guidanceScale = 7.5;

  void setGuidanceScale(value) {
    guidanceScale = value;
  }

  void generate() async {
    if (_formKey.currentState!.validate()) {
      _formKey.currentState!.save();
      var map = new Map<String, dynamic>();
      map["prompt"] = prompt;
      map["negative_prompt"] = negativePrompt;
      map["use_multiple_models"] = useMultipleModels.toString();
      map["use_nsfw"] = useNsfw.toString();
      map["use_random_seed"] = useRandomSeed.toString();
      map["seed"] = seed.toString();
      map["width"] = width.toString();
      map["height"] = height.toString();
      map["inference_steps"] = inferenceSteps.toString();
      map["guidance_scale"] = guidanceScale.toString();
      var ns = NetworkService();
      var response = await ns.get("http://localhost:8000/csrf");
      var document = parse(response);
      var csrf = document.querySelectorAll("input").first.attributes["value"];
      map["csrfmiddlewaretoken"] = csrf;
      ns.headers["X-CSRFToken"] = csrf!;
      response = await ns.post(
        "http://localhost:8000/queue_prompt",
        body: map,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    // Build a Form widget using the _formKey created above.

    return Form(
      key: _formKey,
      child: Flexible(
        child: Row(children: [
          if (useAdvanced) ...[
            AdvancedColumn(
                useRandomSeed: useRandomSeed,
                seed: seed,
                width: width,
                height: height,
                inferenceSteps: inferenceSteps,
                guidanceScale: guidanceScale,
                setUseRandomSeed: setUseRandomSeed,
                setSeed: setSeed,
                setWidth: setWidth,
                setHeight: setHeight,
                setInferenceSteps: setInferenceSteps,
                setGuidanceScale: setGuidanceScale),
            VerticalSeparator(),
            PromptColumn(
              setPrompt: setPrompt,
              setNegativePrompt: setNegativePrompt,
              setUseAdvanced: setUseAdvanced,
              prompt: prompt,
              negativePrompt: negativePrompt,
              generate: generate,
            ),
          ] else ...[
            PromptColumn(
              setPrompt: setPrompt,
              setNegativePrompt: setNegativePrompt,
              setUseAdvanced: setUseAdvanced,
              prompt: prompt,
              negativePrompt: negativePrompt,
              generate: generate,
            ),
          ]
        ]),
      ),
    );
  }
}
