class DentalOntology:
    """Ontology containing dental concepts, relations, and detailed information"""

    def __init__(self):
        self.conditions = {
            "Crown": {
                "definition": "A dental restoration that completely caps or encircles a tooth or dental implant.",
                "types": ["Porcelain", "Metal", "Porcelain-fused-to-metal", "Zirconia", "E-max"],
                "clinical_significance": "Used to restore a damaged tooth's shape, size, strength, and appearance.",
                "related_conditions": ["Root Canal Treatment", "Abutment"],
                "differential_diagnosis": ["Filling", "Veneer"]
            },
            "Implant": {
                "definition": "A surgical component that interfaces with the bone of the jaw to support a dental prosthesis.",
                "types": ["Endosteal", "Subperiosteal", "Zygomatic"],
                "clinical_significance": "Replacement for missing teeth that provides stable support for artificial teeth.",
                "related_conditions": ["Bone Loss", "Abutment", "Crown"],
                "differential_diagnosis": ["Dentures", "Bridges"]
            },
            "Root Piece": {
                "definition": "A fragment of tooth root remaining in the alveolar bone after unsuccessful extraction.",
                "types": ["Apical fragment", "Lateral fragment", "Furcation fragment"],
                "clinical_significance": "May cause infection, pain, or interfere with implant placement if not removed.",
                "related_conditions": ["Periapical lesion", "Retained root"],
                "differential_diagnosis": ["Root Canal Treatment", "Fracture teeth"]
            },
            "Filling": {
                "definition": "A dental restorative material used to restore the function, integrity, and morphology of missing tooth structure.",
                "types": ["Amalgam", "Composite", "Glass Ionomer", "Gold", "Porcelain"],
                "clinical_significance": "Restores tooth structure lost to decay or trauma.",
                "related_conditions": ["Caries", "Fracture teeth"],
                "differential_diagnosis": ["Crown", "Inlay/Onlay"]
            },
            "Periapical lesion": {
                "definition": "A pathological area around the apex (tip) of a tooth root, usually associated with infection or inflammation.",
                "types": ["Periapical granuloma", "Periapical cyst", "Periapical abscess"],
                "clinical_significance": "Indicates infection or inflammation that may require endodontic treatment.",
                "related_conditions": ["Root Canal Treatment", "Caries", "Fracture teeth"],
                "differential_diagnosis": ["Bone defect", "Cyst"]
            },
            "Retained root": {
                "definition": "A root or roots of a tooth that remain in the alveolar bone after the crown has been lost.",
                "types": ["Intentional root retention", "Unintentional root retention"],
                "clinical_significance": "May be asymptomatic but can lead to infection or complicate dental treatment.",
                "related_conditions": ["Root Piece", "Periapical lesion"],
                "differential_diagnosis": ["Root Canal Treatment", "Implant"]
            },
            "maxillary sinus": {
                "definition": "One of the paranasal sinuses, located in the maxillary bone above the upper teeth.",
                "types": ["Normal", "Mucosal thickening", "Sinusitis", "Pneumatization"],
                "clinical_significance": "Proximity to upper molars affects extraction and implant placement.",
                "related_conditions": ["Sinusitis", "Oro-antral communication"],
                "differential_diagnosis": ["Cyst", "Tumor"]
            },
            "Malaligned": {
                "definition": "Teeth that are not properly aligned or positioned in the dental arch.",
                "types": ["Crowding", "Spacing", "Rotation", "Tipping", "Supraversion"],
                "clinical_significance": "Can affect oral hygiene, function, and aesthetics.",
                "related_conditions": ["Impacted tooth", "Supra Eruption"],
                "differential_diagnosis": ["TAD", "Orthodontic brackets"]
            }
        }

    def get_condition_info(self, condition):
        """Return detailed information about a dental condition"""
        return self.conditions.get(condition, {
            "definition": "Information not available for this condition.",
            "types": [],
            "clinical_significance": "",
            "related_conditions": [],
            "differential_diagnosis": []
        })

    def get_related_conditions(self, condition):
        """Return conditions related to the specified condition"""
        info = self.get_condition_info(condition)
        return info.get("related_conditions", [])

    def get_differential_diagnosis(self, condition):
        """Return differential diagnosis for the specified condition"""
        info = self.get_condition_info(condition)
        return info.get("differential_diagnosis", [])