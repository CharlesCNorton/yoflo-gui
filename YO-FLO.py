import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from colorama import Fore, Style, init
import threading
import os
import time
from datetime import datetime
from huggingface_hub import hf_hub_download

init(autoreset=True)


class YO_FLO:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.headless_mode = False
        self.processor = None
        self.inference_phrases_result_labels = []
        self.scaler = torch.cuda.amp.GradScaler()
        self.inference_start_time = None
        self.inference_count = 0
        self.inference_rate_label = None
        self.class_name = None
        self.detections = []
        self.beep_active = False
        self.screenshot_active = False
        self.screenshot_on_yes_active = False
        self.screenshot_on_no_active = False
        self.target_detected = False
        self.last_beep_time = 0
        self.stop_webcam_flag = threading.Event()
        self.model_path = None
        self.phrase = None
        self.debug = False
        self.caption_label = None
        self.object_detection_active = False
        self.expression_comprehension_active = False
        self.visual_grounding_active = False
        self.visual_grounding_phrase = None
        self.webcam_thread = None
        self.inference_title = None
        self.inference_phrases = []
        self.inference_result_label = None
        self.inference_tree_active = False
        self.root = tk.Tk()
        self.root.withdraw()

    def init_model(self, model_path):
        try:
            self.model = (
                AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
                .eval()
                .to(self.device)
                .half()
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model_path = model_path
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Model loaded successfully from {model_path} in fp16{Style.RESET_ALL}"
            )
        except FileNotFoundError:
            print(
                f"{Fore.RED}{Style.BRIGHT}Model path not found: {model_path}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error loading model: {e}{Style.RESET_ALL}")

    def update_inference_rate(self):
        if self.inference_start_time is None:
            self.inference_start_time = time.time()
        else:
            elapsed_time = time.time() - self.inference_start_time
            if elapsed_time > 0:
                inferences_per_second = self.inference_count / elapsed_time
                self.inference_rate_label.config(
                    text=f"Inferences/sec: {inferences_per_second:.2f}", fg="green"
                )

    def toggle_headless(self):
        try:
            self.headless_mode = not self.headless_mode
            status = "enabled" if self.headless_mode else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Headless mode is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling headless mode: {e}{Style.RESET_ALL}"
            )

    def prepare_inputs(self, task_prompt, image, phrase=None):
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(
            self.device
        )
        if phrase:
            inputs["input_ids"] = torch.cat(
                [
                    inputs["input_ids"],
                    self.processor.tokenizer(phrase, return_tensors="pt")
                    .input_ids[:, 1:]
                    .to(self.device),
                ],
                dim=1,
            )
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.half()
        return inputs

    def run_model(self, inputs):
        with torch.amp.autocast("cuda"):
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=1,
            )
        return generated_ids

    def process_object_detection_outputs(self, generated_ids, image_size):
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task="<OD>", image_size=image_size
        )
        return parsed_answer

    def process_expression_comprehension_outputs(self, generated_ids):
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return generated_text

    def run_object_detection(self, image):
        try:
            if not self.model or not self.processor:
                raise ValueError("Model or processor is not initialized.")
            task_prompt = "<OD>"
            if self.debug:
                print(f"Running object detection with task prompt: {task_prompt}")
            inputs = self.prepare_inputs(task_prompt, image)
            generated_ids = self.run_model(inputs)
            if self.debug:
                print(f"Generated IDs: {generated_ids}")
            parsed_answer = self.process_object_detection_outputs(
                generated_ids, image.size
            )
            if self.debug:
                print(f"Parsed answer: {parsed_answer}")
            return parsed_answer
        except AttributeError as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Model or processor not initialized properly: {e}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error running object detection: {e}{Style.RESET_ALL}"
            )

    def run_expression_comprehension(self, image, phrase):
        try:
            task_prompt = "<CAPTION_TO_EXPRESSION_COMPREHENSION>"
            if self.debug:
                print(
                    f"Running expression comprehension with task prompt: {task_prompt} and phrase: {phrase}"
                )
            inputs = self.prepare_inputs(task_prompt, image, phrase)
            generated_ids = self.run_model(inputs)
            if self.debug:
                print(f"Generated IDs: {generated_ids}")
            generated_text = self.process_expression_comprehension_outputs(
                generated_ids
            )
            if self.debug:
                print(f"Generated text: {generated_text}")
            return generated_text
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error running expression comprehension: {e}{Style.RESET_ALL}"
            )

    def evaluate_inference_tree(self, image):
        try:
            if not self.inference_phrases:
                print(
                    f"{Fore.RED}{Style.BRIGHT}No inference phrases set.{Style.RESET_ALL}"
                )
                return "FAIL", []
            results = []
            phrase_results = []
            for phrase in self.inference_phrases:
                result = self.run_expression_comprehension(image, phrase)
                if result:
                    if "yes" in result.lower():
                        results.append(True)
                        phrase_results.append(True)
                    else:
                        results.append(False)
                        phrase_results.append(False)
            overall_result = "PASS" if all(results) else "FAIL"
            return overall_result, phrase_results
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error evaluating inference tree: {e}{Style.RESET_ALL}"
            )
            return "FAIL", []

    def run_visual_grounding(self, image, phrase):
        try:
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            inputs = self.prepare_inputs(task_prompt, image, phrase)
            generated_ids = self.run_model(inputs)
            if self.debug:
                print(f"Generated IDs: {generated_ids}")
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            if self.debug:
                print(f"Generated text: {generated_text}")
            parsed_answer = self.processor.post_process_generation(
                generated_text, task=task_prompt, image_size=image.size
            )
            if self.debug:
                print(f"Parsed answer: {parsed_answer}")
            if task_prompt in parsed_answer and parsed_answer[task_prompt]["bboxes"]:
                return parsed_answer[task_prompt]["bboxes"][0]
            else:
                return None
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error running visual grounding: {e}{Style.RESET_ALL}"
            )

    def plot_bbox(self, image):
        try:
            if not self.detections:
                return image
            for bbox, label in self.detections:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            return image
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error plotting bounding boxes: {e}{Style.RESET_ALL}"
            )

    def plot_visual_grounding_bbox(self, image, bbox, phrase):
        try:
            if bbox:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    image,
                    phrase,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
            return image
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error plotting visual grounding bounding box: {e}{Style.RESET_ALL}"
            )

    def select_model_path(self):
        try:
            root = tk.Tk()
            root.withdraw()
            model_path = filedialog.askdirectory()
            if model_path:
                self.init_model(model_path)
            else:
                print(
                    f"{Fore.YELLOW}{Style.BRIGHT}Model path selection cancelled.{Style.RESET_ALL}"
                )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error selecting model path: {e}{Style.RESET_ALL}"
            )

    def download_model(self):
        try:
            model_name = "microsoft/Florence-2-base-ft"
            model_path = hf_hub_download(
                repo_id=model_name, filename="pytorch_model.bin"
            )
            processor_path = hf_hub_download(
                repo_id=model_name, filename="preprocessor_config.json"
            )
            local_model_dir = os.path.dirname(model_path)
            self.init_model(local_model_dir)
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Model downloaded and initialized from {local_model_dir}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error downloading model: {e}{Style.RESET_ALL}"
            )

    def set_class_name(self):
        try:
            class_name = simpledialog.askstring(
                "Set Class Name",
                "Enter the class name you want to detect (leave blank to show all detections, e.g., 'cat', 'dog'):",
            )
            self.class_name = class_name if class_name else None
            if self.class_name:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Set to detect: {self.class_name}{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Showing all detections{Style.RESET_ALL}"
                )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error setting class name: {e}{Style.RESET_ALL}"
            )

    def set_phrase(self):
        try:
            phrase = simpledialog.askstring(
                "Set Phrase",
                "Enter the yes or no question you want answered (e.g., 'Is the person smiling?', 'Is the cat laying down?'):",
            )
            self.phrase = phrase if phrase else None
            if self.phrase:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Set to comprehend: {self.phrase}{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}No phrase set for comprehension{Style.RESET_ALL}"
                )
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error setting phrase: {e}{Style.RESET_ALL}")

    def set_visual_grounding_phrase(self):
        try:
            phrase = simpledialog.askstring(
                "Set Visual Grounding Phrase", "Enter the phrase for visual grounding:"
            )
            self.visual_grounding_phrase = phrase if phrase else None
            if self.visual_grounding_phrase:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Set visual grounding phrase: {self.visual_grounding_phrase}{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}No phrase set for visual grounding{Style.RESET_ALL}"
                )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error setting visual grounding phrase: {e}{Style.RESET_ALL}"
            )

    def set_inference_tree(self):
        try:
            self.inference_title = simpledialog.askstring(
                "Inference Title", "Enter the title for the inference tree:"
            )
            self.inference_phrases = []
            for i in range(3):
                phrase = simpledialog.askstring(
                    "Set Inference Phrase",
                    f"Enter inference phrase {i+1} (e.g., 'Is it cloudy?', 'Is it wet?'):",
                )
                if phrase:
                    self.inference_phrases.append(phrase)
                else:
                    print(
                        f"{Fore.YELLOW}{Style.BRIGHT}Cancelled setting inference phrase {i+1}.{Style.RESET_ALL}"
                    )
                    return
            if self.inference_title and self.inference_phrases:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Inference tree set with title: {self.inference_title}{Style.RESET_ALL}"
                )
                for phrase in self.inference_phrases:
                    print(
                        f"{Fore.GREEN}{Style.BRIGHT}Inference phrase: {phrase}{Style.RESET_ALL}"
                    )
            else:
                print(
                    f"{Fore.YELLOW}{Style.BRIGHT}Inference tree setting cancelled.{Style.RESET_ALL}"
                )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error setting inference tree: {e}{Style.RESET_ALL}"
            )

    def toggle_beep(self):
        try:
            self.beep_active = not self.beep_active
            status = "active" if self.beep_active else "inactive"
            print(f"{Fore.GREEN}{Style.BRIGHT}Beep is now {status}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error toggling beep: {e}{Style.RESET_ALL}")

    def toggle_screenshot(self):
        try:
            self.screenshot_active = not self.screenshot_active
            status = "active" if self.screenshot_active else "inactive"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Screenshot on detection is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling screenshot: {e}{Style.RESET_ALL}"
            )

    def toggle_screenshot_on_yes(self):
        try:
            self.screenshot_on_yes_active = not self.screenshot_on_yes_active
            status = "active" if self.screenshot_on_yes_active else "inactive"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Screenshot on Yes Inference is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling Screenshot on Yes Inference: {e}{Style.RESET_ALL}"
            )

    def toggle_screenshot_on_no(self):
        try:
            self.screenshot_on_no_active = not self.screenshot_on_no_active
            status = "active" if self.screenshot_on_no_active else "inactive"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Screenshot on No Inference is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling Screenshot on No Inference: {e}{Style.RESET_ALL}"
            )

    def toggle_debug(self):
        try:
            self.debug = not self.debug
            status = "enabled" if self.debug else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Debug mode is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling debug mode: {e}{Style.RESET_ALL}"
            )

    def toggle_object_detection(self):
        try:
            self.object_detection_active = not self.object_detection_active
            if not self.object_detection_active:
                self.detections.clear()
                self.class_name = "null"
                self.update_display()
            status = "enabled" if self.object_detection_active else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Object detection is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling object detection: {e}{Style.RESET_ALL}"
            )

    def toggle_expression_comprehension(self):
        try:
            self.expression_comprehension_active = (
                not self.expression_comprehension_active
            )
            status = "enabled" if self.expression_comprehension_active else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Expression comprehension is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling expression comprehension: {e}{Style.RESET_ALL}"
            )

    def toggle_visual_grounding(self):
        try:
            self.visual_grounding_active = not self.visual_grounding_active
            status = "enabled" if self.visual_grounding_active else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Visual grounding is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling visual grounding: {e}{Style.RESET_ALL}"
            )

    def toggle_inference_tree(self):
        try:
            self.inference_tree_active = not self.inference_tree_active
            status = "enabled" if self.inference_tree_active else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Inference tree evaluation is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling inference tree: {e}{Style.RESET_ALL}"
            )

    def update_caption_window(self, caption):
        if self.caption_label:
            if caption.lower() == "yes":
                self.caption_label.config(
                    text=caption, fg="green", bg="black", font=("Helvetica", 14, "bold")
                )
                if self.screenshot_on_yes_active:
                    self.save_screenshot(
                        cv2.cvtColor(np.array(self.latest_image), cv2.COLOR_RGB2BGR)
                    )
            elif caption.lower() == "no":
                self.caption_label.config(
                    text=caption, fg="red", bg="black", font=("Helvetica", 14, "bold")
                )
                if self.screenshot_on_no_active:
                    self.save_screenshot(
                        cv2.cvtColor(np.array(self.latest_image), cv2.COLOR_RGB2BGR)
                    )
            else:
                self.caption_label.config(
                    text=caption, fg="white", bg="black", font=("Helvetica", 14, "bold")
                )

    def update_inference_result_window(self, result, phrase_results):
        if self.inference_result_label:
            if result.lower() == "pass":
                self.inference_result_label.config(
                    text=result, fg="green", bg="black", font=("Helvetica", 14, "bold")
                )
            else:
                self.inference_result_label.config(
                    text=result, fg="red", bg="black", font=("Helvetica", 14, "bold")
                )
        for idx, phrase_result in enumerate(phrase_results):
            label = self.inference_phrases_result_labels[idx]
            if phrase_result:
                label.config(
                    text=f"Inference {idx+1}: PASS",
                    fg="green",
                    bg="black",
                    font=("Helvetica", 14, "bold"),
                )
            else:
                label.config(
                    text=f"Inference {idx+1}: FAIL",
                    fg="red",
                    bg="black",
                    font=("Helvetica", 14, "bold"),
                )

    def beep_sound(self):
        try:
            if os.name == "nt":
                os.system("echo \a")
            else:
                print("\a")
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error playing beep sound: {e}{Style.RESET_ALL}"
            )

    def save_screenshot(self, frame):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Screenshot saved: {filename}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error saving screenshot: {e}{Style.RESET_ALL}"
            )

    def start_webcam_detection(self):
        if self.webcam_thread and self.webcam_thread.is_alive():
            print(
                f"{Fore.RED}{Style.BRIGHT}Webcam detection is already running.{Style.RESET_ALL}"
            )
            return
        self.stop_webcam_flag.clear()
        self.webcam_thread = threading.Thread(target=self._webcam_detection_thread)
        self.webcam_thread.start()

    def _webcam_detection_thread(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print(
                    f"{Fore.RED}{Style.BRIGHT}Error: Could not open webcam.{Style.RESET_ALL}"
                )
                return
            while not self.stop_webcam_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(
                        f"{Fore.RED}{Style.BRIGHT}Error: Failed to capture image from webcam.{Style.RESET_ALL}"
                    )
                    break
                try:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image)
                    self.latest_image = image_pil
                    if self.debug:
                        print(f"Captured frame from webcam")
                    if self.expression_comprehension_active and self.phrase:
                        if self.debug:
                            print(
                                f"Expression comprehension enabled with phrase: {self.phrase}"
                            )
                        results = self.run_expression_comprehension(
                            image_pil, self.phrase
                        )
                        if results:
                            caption = "Yes" if "yes" in results.lower() else "No"
                            self.update_caption_window(caption)
                            if self.headless_mode:
                                print(f"Expression comprehension result: {caption}")
                            self.inference_count += 1
                            self.update_inference_rate()
                    if self.object_detection_active:
                        if self.debug:
                            print(f"Running object detection")
                        results = self.run_object_detection(image_pil)
                        if results and "<OD>" in results:
                            self.target_detected = False
                            self.detections = []
                            for bbox, label in zip(
                                results["<OD>"]["bboxes"], results["<OD>"]["labels"]
                            ):
                                if (
                                    self.class_name is None
                                    or label.lower() == self.class_name.lower()
                                ):
                                    self.detections.append((bbox, label))
                                    if (
                                        self.class_name
                                        and label.lower() == self.class_name.lower()
                                    ):
                                        self.target_detected = True
                            if self.headless_mode:
                                print(f"Object Detection results: {self.detections}")
                            self.inference_count += 1
                            self.update_inference_rate()
                    if self.visual_grounding_active and self.visual_grounding_phrase:
                        if self.debug:
                            print(
                                f"Running visual grounding with phrase: {self.visual_grounding_phrase}"
                            )
                        bbox = self.run_visual_grounding(
                            image_pil, self.visual_grounding_phrase
                        )
                        if bbox:
                            if not self.headless_mode:
                                frame = self.plot_visual_grounding_bbox(
                                    frame, bbox, self.visual_grounding_phrase
                                )
                            else:
                                print(f"Visual Grounding result: {bbox}")
                            self.inference_count += 1
                            self.update_inference_rate()
                    if (
                        self.inference_tree_active
                        and self.inference_title
                        and self.inference_phrases
                    ):
                        inference_result, phrase_results = self.evaluate_inference_tree(
                            image_pil
                        )
                        self.update_inference_result_window(
                            inference_result, phrase_results
                        )
                        if self.headless_mode:
                            print(
                                f"Inference Tree result: {inference_result}, Details: {phrase_results}"
                            )
                        self.inference_count += 1
                        self.update_inference_rate()
                    if not self.headless_mode:
                        bbox_image = self.plot_bbox(frame.copy())
                        cv2.imshow("Object Detection", bbox_image)

                        current_time = time.time()
                        if (
                            self.beep_active
                            and self.target_detected
                            and current_time - self.last_beep_time > 1
                        ):
                            threading.Thread(target=self.beep_sound).start()
                            if self.debug:
                                print(
                                    f"{Fore.GREEN}{Style.BRIGHT}Target detected: {self.class_name}{Style.RESET_ALL}"
                                )
                            self.last_beep_time = current_time
                        if self.screenshot_active and self.target_detected:
                            self.save_screenshot(bbox_image)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    else:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                except Exception as e:
                    print(
                        f"{Fore.RED}{Style.BRIGHT}Error during frame processing: {e}{Style.RESET_ALL}"
                    )
            cap.release()
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"{Fore.RED}{Style.BRIGHT}OpenCV error: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error during webcam detection: {e}{Style.RESET_ALL}"
            )
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.stop_webcam_flag.clear()

    def stop_webcam_detection(self):
        if not self.webcam_thread or not self.webcam_thread.is_alive():
            print(
                f"{Fore.RED}{Style.BRIGHT}Webcam detection is not running.{Style.RESET_ALL}"
            )
            return
        self.object_detection_active = False
        self.expression_comprehension_active = False
        self.visual_grounding_active = False
        self.inference_tree_active = False

        self.update_display()

        self.stop_webcam_flag.set()

        self.root.after(100, self._wait_for_thread_to_stop)

    def _wait_for_thread_to_stop(self):
        if self.webcam_thread.is_alive():
            self.root.after(100, self._wait_for_thread_to_stop)
        else:
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Webcam detection stopped successfully.{Style.RESET_ALL}"
            )

    def update_display(self):
        if not self.object_detection_active:
            empty_frame = np.zeros((480, 640, 3), np.uint8)
            cv2.imshow("Object Detection", empty_frame)
            cv2.waitKey(1)

    def main_menu(self):
        self.root.deiconify()
        self.root.title("YO-FLO Menu")

        def on_closing():
            self.stop_webcam_detection()
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)

        try:
            model_frame = tk.LabelFrame(self.root, text="Model Management")
            model_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                model_frame, text="Select Model Path", command=self.select_model_path
            ).pack(fill="x")
            tk.Button(
                model_frame,
                text="Download Model from HuggingFace",
                command=self.download_model,
            ).pack(fill="x")

            detection_frame = tk.LabelFrame(self.root, text="Detection Settings")
            detection_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                detection_frame,
                text="Set Classes for Object Detection",
                command=self.set_class_name,
            ).pack(fill="x")
            tk.Button(
                detection_frame,
                text="Set Phrase for Yes/No Inference",
                command=self.set_phrase,
            ).pack(fill="x")
            tk.Button(
                detection_frame,
                text="Set Grounding Phrase",
                command=self.set_visual_grounding_phrase,
            ).pack(fill="x")
            tk.Button(
                detection_frame,
                text="Set Inference Tree",
                command=self.set_inference_tree,
            ).pack(fill="x")

            toggle_features_frame = tk.LabelFrame(self.root, text="Toggle Features")
            toggle_features_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                toggle_features_frame,
                text="Toggle Object Detection",
                command=self.toggle_object_detection,
            ).pack(fill="x")
            tk.Button(
                toggle_features_frame,
                text="Toggle Yes/No Inference",
                command=self.toggle_expression_comprehension,
            ).pack(fill="x")
            tk.Button(
                toggle_features_frame,
                text="Toggle Visual Grounding",
                command=self.toggle_visual_grounding,
            ).pack(fill="x")
            tk.Button(
                toggle_features_frame,
                text="Toggle Inference Tree",
                command=self.toggle_inference_tree,
            ).pack(fill="x")
            tk.Button(
                toggle_features_frame,
                text="Toggle Headless Mode",
                command=self.toggle_headless,
            ).pack(fill="x")

            toggle_triggers_frame = tk.LabelFrame(self.root, text="Toggle Triggers")
            toggle_triggers_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                toggle_triggers_frame,
                text="Toggle Beep on Detection",
                command=self.toggle_beep,
            ).pack(fill="x")
            tk.Button(
                toggle_triggers_frame,
                text="Toggle Screenshot on Detection",
                command=self.toggle_screenshot,
            ).pack(fill="x")
            tk.Button(
                toggle_triggers_frame,
                text="Toggle Screenshot on Yes Inference",
                command=self.toggle_screenshot_on_yes,
            ).pack(fill="x")
            tk.Button(
                toggle_triggers_frame,
                text="Toggle Screenshot on No Inference",
                command=self.toggle_screenshot_on_no,
            ).pack(fill="x")

            webcam_frame = tk.LabelFrame(self.root, text="Webcam Control")
            webcam_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                webcam_frame,
                text="Start Webcam Detection",
                command=self.start_webcam_detection,
            ).pack(fill="x")
            tk.Button(
                webcam_frame,
                text="Stop Webcam Detection",
                command=self.stop_webcam_detection,
            ).pack(fill="x")

            tk.Button(
                self.root, text="Toggle Debug Mode", command=self.toggle_debug
            ).pack(fill="x", padx=10, pady=10)

            inference_rate_frame = tk.LabelFrame(self.root, text="Inference Rate")
            inference_rate_frame.pack(fill="x", padx=10, pady=5)
            self.inference_rate_label = tk.Label(
                inference_rate_frame,
                text="Inferences/sec: N/A",
                fg="white",
                bg="black",
                font=("Helvetica", 14, "bold"),
            )
            self.inference_rate_label.pack(fill="x")

            binary_inference_frame = tk.LabelFrame(self.root, text="Binary Inference")
            binary_inference_frame.pack(fill="x", padx=10, pady=5)
            self.caption_label = tk.Label(
                binary_inference_frame,
                text="Binary Inference: N/A",
                fg="white",
                bg="black",
                font=("Helvetica", 14, "bold"),
            )
            self.caption_label.pack(fill="x")

            inference_tree_frame = tk.LabelFrame(self.root, text="Inference Tree")
            inference_tree_frame.pack(fill="x", padx=10, pady=5)
            self.inference_result_label = tk.Label(
                inference_tree_frame,
                text="Inference Tree: N/A",
                fg="white",
                bg="black",
                font=("Helvetica", 14, "bold"),
            )
            self.inference_result_label.pack(fill="x")

            for i in range(3):
                label = tk.Label(
                    inference_tree_frame,
                    text=f"Inference {i+1}: N/A",
                    fg="white",
                    bg="black",
                    font=("Helvetica", 14, "bold"),
                )
                label.pack(fill="x")
                self.inference_phrases_result_labels.append(label)
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error creating menu: {e}{Style.RESET_ALL}")
        self.root.mainloop()


if __name__ == "__main__":
    try:
        yo_flo = YO_FLO()
        print(
            f"{Fore.BLUE}{Style.BRIGHT}Discover YO-FLO: A proof-of-concept in using advanced vision models as a YOLO alternative.{Style.RESET_ALL}"
        )
        yo_flo.main_menu()
    except Exception as e:
        print(
            f"{Fore.RED}{Style.BRIGHT}Error initializing YO-FLO: {e}{Style.RESET_ALL}"
        )
