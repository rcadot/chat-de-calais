# doc_reader.py
"""Extraction du texte des fichiers Word 97-2003 (.doc), en Python pur.

Ni LibreOffice ni Word ne sont nécessaires : le texte est lu directement dans
le conteneur OLE à l'aide de la table des pièces (piece table) décrite dans la
spécification [MS-DOC].

Principe :
- le flux `WordDocument` commence par le FIB (File Information Block), qui
  indique quel flux de table utiliser (`0Table` ou `1Table`), la position de
  la structure CLX dans ce flux et le nombre de caractères du corps du texte ;
- la CLX contient la PlcPcd : une suite de « pièces », chacune décrivant un
  morceau de texte stocké soit en cp1252 (compressé), soit en UTF-16LE ;
- le texte brut contient des caractères de contrôle (fin de paragraphe,
  cellules, champs) que l'on convertit ou supprime.
"""

import re
import struct

# Positions dans le FIB (Word 97 et ultérieurs)
_FIB_FLAGS = 0x000A  # FibBase : fWhichTblStm (0x0200), fEncrypted (0x0100)
_FIB_CCP_TEXT = 0x004C  # FibRgLw97.ccpText : caractères du corps
_FIB_CCP_FTN = 0x0050  # FibRgLw97.ccpFtn : caractères des notes de bas de page
_FIB_FC_CLX = 0x01A2  # FibRgFcLcb97.fcClx / lcbClx

_FIELD_BEGIN, _FIELD_SEP, _FIELD_END = "\x13", "\x14", "\x15"


def _read_pieces(word: bytes, table: bytes) -> str:
    """Reconstitue le texte brut à partir de la table des pièces."""
    fc_clx, lcb_clx = struct.unpack_from("<II", word, _FIB_FC_CLX)
    clx = table[fc_clx : fc_clx + lcb_clx]

    pos = 0
    while pos < len(clx) and clx[pos] == 0x01:  # Prc : modifications de propriétés
        (cb_grpprl,) = struct.unpack_from("<H", clx, pos + 1)
        pos += 3 + cb_grpprl
    if pos >= len(clx) or clx[pos] != 0x02:
        raise ValueError("Structure CLX introuvable (fichier .doc non standard)")

    (lcb,) = struct.unpack_from("<I", clx, pos + 1)
    plc = clx[pos + 5 : pos + 5 + lcb]
    n_pieces = (lcb - 4) // 12
    cps = struct.unpack_from(f"<{n_pieces + 1}I", plc, 0)

    parts = []
    for i in range(n_pieces):
        (fc,) = struct.unpack_from("<I", plc, 4 * (n_pieces + 1) + 8 * i + 2)
        length = cps[i + 1] - cps[i]
        if fc & 0x40000000:  # texte « compressé » : un octet par caractère, cp1252
            start = (fc & ~0x40000000) // 2
            parts.append(word[start : start + length].decode("cp1252", errors="replace"))
        else:
            parts.append(word[fc : fc + 2 * length].decode("utf-16-le", errors="replace"))
    return "".join(parts)


def _strip_fields(text: str) -> str:
    """
    Supprime le code des champs Word en gardant leur résultat affiché.

    Un champ s'écrit \\x13 code \\x14 résultat \\x15 (le séparateur et le
    résultat sont facultatifs) ; les champs peuvent être imbriqués.
    """
    out = []
    stack = []  # pour chaque champ ouvert : True si l'on est dans sa partie code
    for char in text:
        if char == _FIELD_BEGIN:
            stack.append(True)
        elif char == _FIELD_SEP and stack:
            stack[-1] = False
        elif char == _FIELD_END and stack:
            stack.pop()
        elif not any(stack):
            out.append(char)
    return "".join(out)


def _clean(text: str) -> str:
    """Convertit les caractères de contrôle Word en texte lisible."""
    text = _strip_fields(text)
    text = text.replace("\x07\x07", "\n")  # fin de cellule suivie de la marque de fin de ligne
    text = text.replace("\x07", "\t")  # fin de cellule
    text = re.sub(r"[\r\x0b\x0c\x0e]", "\n", text)  # paragraphe, saut de ligne, de page, de section
    text = text.replace("\x1e", "-").replace("\x1f", "")  # trait d'union insécable, conditionnel
    text = re.sub(r"[\x00-\x08\x10-\x1d]", "", text)  # images, objets, autres contrôles
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_doc_text(file_path: str) -> str:
    """
    Extrait le texte (corps et notes de bas de page) d'un fichier .doc.

    Args:
        file_path: Chemin du fichier Word 97-2003

    Returns:
        str: Texte du document

    Raises:
        ValueError: Fichier chiffré ou structure non reconnue
        OSError: Fichier illisible ou qui n'est pas un conteneur OLE
    """
    import olefile

    if not olefile.isOleFile(file_path):
        raise ValueError("Ce fichier .doc n'est pas un document Word 97-2003 (conteneur OLE)")

    with olefile.OleFileIO(file_path) as ole:
        word = ole.openstream("WordDocument").read()
        (flags,) = struct.unpack_from("<H", word, _FIB_FLAGS)
        if flags & 0x0100:
            raise ValueError("Document Word chiffré (protégé par mot de passe)")
        table_name = "1Table" if flags & 0x0200 else "0Table"
        table = ole.openstream(table_name).read()

    ccp_text, ccp_ftn = struct.unpack_from("<ii", word, _FIB_CCP_TEXT)
    raw = _read_pieces(word, table)
    return _clean(raw[: max(0, ccp_text) + max(0, ccp_ftn)])
