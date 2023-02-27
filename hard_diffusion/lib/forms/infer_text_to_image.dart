
// Create a Form widget.
import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/prompts.dart';

class InferTextToImageForm extends StatefulWidget {
  const InferTextToImageForm({super.key});

  @override
  InferTextToImageFormState createState() {
    return InferTextToImageFormState();
  }
}

// Create a corresponding State class.
// This class holds data related to the form.
class InferTextToImageFormState extends State<InferTextToImageForm> {
  // Create a global key that uniquely identifies the Form widget
  // and allows validation of the form.
  //
  // Note: This is a GlobalKey<FormState>,
  // not a GlobalKey<InferTextToImageFormState>.
  final _formKey = GlobalKey<FormState>();

  @override
  Widget build(BuildContext context) {
    // Build a Form widget using the _formKey created above.
    return Form(
      key: _formKey,
      child: Column(
        children: [
          PromptField(),
          NegativePromptField(),
        ],
      ),
    );
  }
}
