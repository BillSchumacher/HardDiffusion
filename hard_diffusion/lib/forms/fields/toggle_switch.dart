import 'package:flutter/material.dart';

class ToggleSwitch extends StatefulWidget {
  const ToggleSwitch(
      {super.key,
      required this.value,
      required this.setValue,
      required this.label});

  final bool value;
  final Function(bool) setValue;
  final String label;
  @override
  State<ToggleSwitch> createState() => _ToggleSwitchState(
        value: value,
        setValue: setValue,
        label: label,
      );
}

class _ToggleSwitchState extends State<ToggleSwitch> {
  _ToggleSwitchState(
      {required this.value, required this.setValue, required this.label});
  bool value;
  Function(bool) setValue;
  String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(left: 2, right: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Switch(
            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
            splashRadius: 0,
            value: value,
            activeColor: Colors.orange,
            onChanged: (bool newValue) {
              setState(() {
                value = newValue;
                setValue(newValue);
              });
            },
          ),
        ],
      ),
    );
  }
}
