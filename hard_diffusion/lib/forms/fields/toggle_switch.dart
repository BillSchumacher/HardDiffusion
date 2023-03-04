import 'package:flutter/material.dart';

class ToggleSwitch extends StatefulWidget {
  const ToggleSwitch({super.key, required this.setValue, required this.label});

  final Function(bool) setValue;
  final String label;
  @override
  State<ToggleSwitch> createState() => _ToggleSwitchState(
        setValue: setValue,
        label: label,
      );
}

class _ToggleSwitchState extends State<ToggleSwitch> {
  _ToggleSwitchState({required this.setValue, required this.label});
  bool light = false;

  Function(bool) setValue;
  String label;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label),
        Switch(
          value: light,
          activeColor: Colors.orange,
          onChanged: (bool value) {
            setState(() {
              light = value;
            });
            setValue(light);
          },
        ),
      ],
    );
  }
}
