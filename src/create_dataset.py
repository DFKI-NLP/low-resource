import json
import gzip
from collections import namedtuple
from typing import List, Dict, Any
from src.pipeline.create_pipline_init import create_nlp_pipeline
from spacy.language import Language

Instance = namedtuple("Instance", ["id", "text"])
AnnotatedInstance = namedtuple("AnnotatedInstance", ["id", "docid", "token", "ner", "pos", "dep_head", "dep_rel"])


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def instances_from_amazon() -> List[Instance]:
    path = "/data2/zhanghc/data_external/amazon_reviews/meta_Electronics.json.gz"
    i = 0
    instances = []
    for example in parse(path):
        i += 1
        for cat in example['categories'][0]:
            if "Computer" in cat and "description" in example:
                instances.append(Instance(id=example["asin"],
                                          text=example["description"]))
                continue

    return instances


def create_annotated_dataset(instances: List[Instance], nlp: Language, output_path: str) -> None:
    annotated_instances = []
    for instance in instances:
        doc = nlp(instance.text)

        if len(doc.ents) == 0:
            continue

        annotated_instances.append(AnnotatedInstance(id=instance.id,
                                                     docid=instance.id,
                                                     token=[t.text for t in doc],
                                                     ner=[t.ent_type_ if t.ent_type_ else "O" for t in doc],
                                                     pos=[t.pos_ for t in doc],
                                                     dep_head=[t.head.i for t in doc],
                                                     dep_rel=[t.dep_ for t in doc]))

    with open(output_path, "w") as out_f:
        json.dump([instance._asdict() for instance in annotated_instances], out_f)

    return annotated_instances


GAZETTEERS_PATH = "/data2/zhanghc/RE/low-resource/src/data/gazetteers/"
nlp = create_nlp_pipeline(gazetteers_path=GAZETTEERS_PATH)

# text ="""Suitable for anyone who needs a powerful, versatile mobile computing solution but ideal for those who appreciate the latest multimedia and home entertainment technologies, the Sony VAIO PCG-GRT360ZG Notebook PC is a beefy, comprehensive rig that will outperform and out-convenience most desktops. Sporting desirable perks such as speedy Intel Hyper-Threading processing, DVD burning, advanced television functions and a handy wireless remote, the VAIO PCG-GRT360ZG is ready for virtually any circumstance.At 14.1 by 11.9 by 1.9 inches and 7.72 pounds (with one battery) and 9.1 pounds (with battery and optical drive), the VAIO PCG-GRT360ZG cannot be considered a true lightweight. It therefore may not be the best notebook for everyday transport. However, if you prefer processing punch to portability, the unit is ready to oblige. Sporting a commanding 3.06 GHz Intel Pentium 4 CPU with Hyper-Threading technology for more efficient data throughput, a fast 533 front side bus and 512 MB DDR SDRAM memory, the VAIO PCG-GRT360ZG will easily power through any application. Sony has complemented this potent data processing "engine" with an impressive 64 MB nVidia GeForce FX Go 5600 3-D video accelerator to make quick work of graphics-intensive 3-D applications and all but the most demanding 3-D games. The unit's integrated 80 GB hard disk offers oodles of room to store important files and programs, and its detachable DVD+RW/CD-RW combo drive allows you to turn home videos into DVDs, watch the latest Hollywood releases, burn and play audio CDs and backup data.Multimedia enthusiasts will appreciate the system's integrated stereo speakers and generous 16.1-inch SXGA+ viewing screen for its image-enhancing X-Brite technology and super-clean 1400 by 1050 maximum resolution. Other desirable amenities include a Memory Stick PRO media slot, three high-speed USB 2.0 ports, a fast IEEE 1394 port for data transfers from peripherals such as digital camcorders, and a cord-free television-style remote. To communicate with the outside world, the unit features a 56K modem for low-speed Internet and email access, a standard 10BASE-T/100BASE-TX Ethernet interface for high-speed connectivity, and a convenient IEEE 802.11b/802.11g wireless LAN.Bundled software includes Microsoft Windows XP Home Edition, Sony's Giga Pocket personal video recording application (for simultaneous television viewing and video recording), and a truly expansive array of audio, video and photo utilities. Battery life is not one of the system's strong points -- it delivers 1.0 to 1.5 hours of usage with one battery attached and 2.0 to 3.0 hours with an optional second battery installed in the multi-purpose bay."""
# texta="""Amazingly small and light, 10 lbs the Sony Vaio PCG-Z1WA notebook PC is ready to hit the road with its bright 14.1-inch Screen, 1.7 GHz battery-saving Centrino processor (for a battery life of up to 6.5 hours), and built-in 54g wireless LAN for connecting to your e-mail and the Web as you roam the hotspots."""

# doc = nlp(text)
# print([(ent.text, ent.label_) for ent in doc.ents])
# displacy.render(doc, style="ent", jupyter=True)


instances = instances_from_amazon()
annotated_instances = create_annotated_dataset(instances, nlp, output_path="data/train_distant.json")
