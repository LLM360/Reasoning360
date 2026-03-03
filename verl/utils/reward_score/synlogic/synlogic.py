import os
import sys
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from .game_of_24.scripts.game_of_24_verifier import GameOf24Verifier
# from .cryptarithm.scripts.cryptarithm_verifier import CryptarithmVerifier
# from .survo.scripts.survo_verifier import SurvoVerifier
from .campsite_verifier import CampsiteVerifier
from .skyscraper_puzzle_verifier import SkyscraperPuzzleVerifier
from .web_of_lies_verifier import WebOfLiesVerifier
from .goods_exchange_verifier import GoodsExchangeVerifier
# from .sudoku.scripts.sudoku_verifier import SudokuVerifier
# from corpus.misc.tasks.zebra_puzzle.scripts.zebra_puzzle_verifier import ZebraPuzzleVerifier
# from corpus.misc.tasks.bbeh.scripts.bbeh_verifier import BBEHVerifier
# from corpus.misc.tasks.arc_agi.scripts.arc_agi_verifier import ArcAGIVerifier
from .object_properties_verifier import ObjectPropertiesVerifier
from .object_counting_verifier import ObjectCountingVerifier
from .star_placement_puzzle_verifier import StarPlacementPuzzleVerifier
from .arrow_maze_verifier import ArrowMazeVerifier
# from .kukurasu.scripts.kukurasu_verifier import KukurasuVerifier
from .number_wall_verifier import NumberWallVerifier
from .numbrix_verifier import NumbrixVerifier
from .norinori_verifier import NorinoriVerifier
from .minesweeper_verifier import MinesweeperVerifier
from .operation_verifier import OperationVerifier
from .word_sorting_mistake_verifier import WordSortingMistakeVerifier
from .math_path_verifier import MathPathVerifier
from .boolean_expressions_verifier import BooleanExpressionsVerifier
from .space_reasoning_verifier import SpaceReasoningVerifier
from .space_reasoning_tree_verifier import SpaceReasoningTreeVerifier
from .word_sorting_verifier import WordSortingVerifier
# from corpus.misc.tasks.gpqa.scripts.gpqa_verifier import GPQAVerifier
# from .cipher.scripts.cipher_verifier import CipherVerifier
from .time_sequence_verifier import TimeSequenceVerifier
from .wordscapes_verifier import WordscapesVerifier
# from corpus.misc.tasks.bbh.scripts.boolean_expressions_verifier import BBHBooleanExpressionsVerifier
# from corpus.misc.tasks.bbh.scripts.causal_judgement_verifier import BBHCausalJudgementVerifier # yes no
# from corpus.misc.tasks.bbh.scripts.date_understanding_verifier import BBHDateUnderstandingVerifier # multi-choice
# from corpus.misc.tasks.bbh.scripts.dyck_languages_verifier import BBHDyckLanguagesVerifier
# from corpus.misc.tasks.bbh.scripts.formal_fallacies_verifier import BBHFormalFallaciesVerifier
# from corpus.misc.tasks.bbh.scripts.multistep_arithmetic_two_verifier import BBHMultistepArithmeticVerifier # number
# from corpus.misc.tasks.bbh.scripts.sports_understanding_verifier import BBHSportsUnderstandingVerifier
# from corpus.misc.tasks.bbh.scripts.web_of_lies_verifier import BBHWebOfLiesVerifier
# from corpus.misc.tasks.bbh.scripts.word_sorting_verifier import BBHWordSortingVerifier
from .game_of_buggy_tables_verifier import BuggyTableVerifier
# from .calcudoko.scripts.calcudoko_verifier import CalcudokoVerifier
from .dyck_language_verifier import DyckLanguageVerifier
from .dyck_language_errors_verifier import DyckLanguageErrorsVerifier
from .dyck_language_reasoning_errors_verifier import DyckLanguageReasoningErrorsVerifier
# from .futoshiki.scripts.futoshiki_verifier import FutoshikiVerifier

# NOTE: Add new tasks in alphabetical order
verifier_classes = {
    "arrow_maze": ArrowMazeVerifier,
    "boolean_expressions": BooleanExpressionsVerifier,
    "buggy_tables": BuggyTableVerifier,
    # "calcudoko": CalcudokoVerifier,
    "campsite": CampsiteVerifier,
    # "cipher": CipherVerifier,
    # "cryptarithm": CryptarithmVerifier,
    "dyck_language": DyckLanguageVerifier,
    "dyck_language_errors": DyckLanguageErrorsVerifier,
    "dyck_language_reasoning_errors": DyckLanguageReasoningErrorsVerifier,
    # "futoshiki": FutoshikiVerifier,
    "goods_exchange": GoodsExchangeVerifier,
    # "gpqa_diamond": GPQAVerifier,
    # "kukurasu": KukurasuVerifier,
    "math_path": MathPathVerifier,
    # "arc_agi": ArcAGIVerifier,
    # "arc_agi_2": ArcAGIVerifier,
    # "mathador": GameOf24Verifier,
    "minesweeper": MinesweeperVerifier,
    "norinori": NorinoriVerifier,
    "number_wall": NumberWallVerifier,
    "numbrix": NumbrixVerifier,
    "object_counting": ObjectCountingVerifier,
    "object_properties": ObjectPropertiesVerifier,
    "operation": OperationVerifier,
    "skyscraper_puzzle": SkyscraperPuzzleVerifier,
    "space_reasoning": SpaceReasoningVerifier,
    "space_reasoning_tree": SpaceReasoningTreeVerifier,
    "star_placement_puzzle": StarPlacementPuzzleVerifier,
    # "sudoku": SudokuVerifier,
    # "survo": SurvoVerifier,
    "time_sequence": TimeSequenceVerifier,
    "web_of_lies": WebOfLiesVerifier,
    "word_sorting": WordSortingVerifier,
    "word_sorting_mistake": WordSortingMistakeVerifier,
    "wordscapes": WordscapesVerifier,
    # "zebra_puzzle": ZebraPuzzleVerifier,
    # ** bbeh_classes,
    # ** bbh_classes,
}